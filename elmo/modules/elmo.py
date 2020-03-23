# 아직 코드 리뷰안한 코드!
from typing import Optional, Tuple, List, Callable, Union

import h5py
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence, pack_padded_sequence
from torch.autograd import Variable

from .encoder_base import _EncoderBase
from .lstm_cell_with_projection import LstmCellWithProjection
from .util import get_lengths_from_binary_sequence_mask, sort_batch_by_length

# We have two types here for the state, because storing the state in something
# which is Iterable (like a tuple, below), is helpful for internal manipulation
# - however, the states are consumed as either Tensors or a Tuple of Tensors, so
# returning them in this format is unhelpful.
RnnState = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]  # pylint: disable=invalid-name
RnnStateStorage = Tuple[torch.Tensor, ...]  # pylint: disable=invalid-name


class _EncoderBase(nn.Module):
    # pyling: disable=abstract-method
    """
    This abstract class serves as a base for the 3 ``Encoder`` abstractions in AllenNLP.
    - :class:`~allennlp.modules.seq2seq_encoders.Seq2SeqEncoders`
    - :class:`~allennlp.modules.seq2vec_encoders.Seq2VecEncoders`
    Additionally, this class provides functionality for sorting sequences by length
    so they can be consumed by Pytorch RNN classes, which require their inputs to be
    sorted by length. Finally, it also provides optional statefulness to all of it's
    subclasses by allowing the caching and retrieving of the hidden states of RNNs.
    """
    def __init__(self, stateful: bool = False) -> None:
        super(_EncoderBase, self).__init__()
        self.stateful = stateful
        self._states: Optional[RnnStateStorage] = None

    def sort_and_run_forward(self,
                             module: Callable[[PackedSequence, Optional[RnnState]],
                                              Tuple[Union[PackedSequence, torch.Tensor], RnnState]],
                             inputs: torch.Tensor,
                             mask: torch.Tensor,
                             hidden_state: Optional[RnnState] = None):
        """
        Pytorch RNNs는 input이 passing되기 전에 정렬되있어야 함
        Seq2xxxEncoders가 이러한 기능을 모두 사용하기에 base class로 제공
        """
        # In some circumstances you may have sequences of zero length. ``pack_padded_sequence``
        # requires all sequence lengths to be > 0, so remove sequences of zero length before
        # calling self._module, then fill with zeros.

        # First count how many sequences are empty.
        batch_size = mask.size(0)
        num_valid = torch.sum(mask[:, 0]).int().item()

        sequence_lengths = get_lengths_from_binary_sequence_mask(mask)
        sorted_inputs, sorted_sequence_lengths, restoration_indices, sorting_indices = \
            sort_batch_by_length(inputs, sequence_lengths)

        # Now create a PackedSequence with only the non-empty, sorted sequences.
        # pad token 제외, 유의미한 값들만 packing
        packed_sequence_input = pack_padded_sequence(sorted_inputs[:num_valid, :, :],
                                                     sorted_sequence_lengths[:num_valid].data.tolist(),
                                                     batch_first=True)

        # Prepare teh initial states.
        if not self.stateful:
            if hidden_state == None:
                initial_states = hidden_state
            elif isinstance(hidden_state, tuple):
                initial_states = [state.index_select(1, sorting_indices)[:, :num_valid, :]
                                  for state in hidden_state]
            else:
                initial_states = self._get_initial_states(batch_size, num_valid, sorting_indices)
        else:
            initial_states = self._get_initial_states(batch_size, num_valid, sorting_indices)

        # Actually call the module on the sorted PackedSequence
        module_output, final_states = module(packed_sequence_input, initial_states)

        return module_output, final_states, restoration_indices

    def _get_initial_states(self,
                            batch_size: int,
                            num_valid: int,
                            sorting_indices: torch.LongTensor) -> Optional[RnnState]:
        """
        RNN의 초기 상태를 반환
        추가적으로, 이 메서드는 batch의 새로운 요소의 초기 상태를 추가하기 위해 상태를 변경하여(mutate)
            호출시 batch size를 처리
        또한 이 메서드는
            1. 배치의 요소 seq. length로 상태를 정렬하는 것과
            2. pad가 끝난 row 제거도 처리
        중요한 것은 현재의 배치 크기가 이전에 호출되었을 때보다 더 크면 이 상태를 "혼합"하는 것이다.

        이 메서드는 (1) 처음 호출되어 아무 상태가 없는 경우 (2) RNN이 heterogeneous state를 가질 때
        의 경우를 처리해야 하기 때문에 return값이 복잡함

        (1) module이 처음 호출됬을 때 ``module``의 타입이 무엇이든 ``None`` 반환
        (2) Otherwise,
            - LSTM의 경우 tuple of ``torch.Tensor``
              shape: ``(num_layers, num_valid, state_size)``
                 and ``(num_layers, num_valid, memory_size)``
            - GRU의 경우  single ``torch.Tensor``
              shape: ``(num_layers, num_valid, state_size)``
        """
        # We don't know the state sizes the first time calling forward,
        # so we let the module define what it's initial hidden state looks like.
        if self._states is None:
            return None

        # Otherwise, we have some previous states.
        if batch_size > self._states[0].size(1):
            # This batch is larger than the all previous states.
            # If so, resize the states.
            num_states_to_concat = batch_size - self._states[0].size(1)
            resized_states = []
            # state has shape (num_layers, batch_size, hidden_size)
            for state in self._states:
                # This _must_ be inside the loop because some
                # RNNs have states with different last dimension sizes.
                zeros = state.data.new(state.size(0),
                                       num_states_to_concat,
                                       state.size(2)).fill_(0)
                zeros = Variable(zeros)
                resized_states.append(torch.cat([state, zeros], 1))
            self._states = tuple(resized_states)
            correctly_shaped_states = self._states
        elif batch_size < self._states[0].size(1):
            # This batch is smaller than the previous one.
            correctly_shaped_states = tuple(staet[:, :batch_size, :] for state in self._states)
        else:
            correctly_shaped_states = self._states

        # At this point, out states are of shape (num_layers, batch_size, hidden_size).
        # However, the encoder uses sorted sequences and additionally removes elements
        # of the batch which are fully padded. We need the states to match up to these
        # sorted and filtered sequences, so we do that in the next two blocks before
        # returning the states.
        if len(self._states) == 1:
            # GRU
            correctly_shaped_state = correctly_shaped_states[0]
            sorted_state = correctly_shaped_state.index_select(1, sorting_indices)
            return sorted_state[:, :num_valid, :]
        else:
            # LSTM
            sorted_states = [state.index_select(1, sorting_indices)
                             for state in correctly_shaped_states]
            return tuple(state[:, :num_valid, :] for state in sorted_states)

    def _update_states(self,
                       final_states: RnnStateStorage,
                       restoration_indices: torch.LongTensor) -> None:
        """
        RNN forward 동작 후에 state를 update
        새로운 state로 update하며 몇 가지 book-keeping을 실시
        즉, 상태를 해제하고 완전히 padding된 state가 업데이트되지 않도록 함
        마지막으로 graph가 매 batch iteration후에 gc되도록 계산 그래프에서
        state variable을 떼어냄.
        """
        # TODO(Mark)L seems weird to sort here, but append zeros in the subclasses.
        # which way around is best?
        new_unsorted_states = [state.index_select(1, restoration_indices)
                               for state in final_states]

        if self._states is None:
            # We don't already have states, so just set the
            # ones we receive to be the current state.
            self._states = tuple([Variable(state.data)
                                  for state in new_unsorted_states])
        else:
            current_state_batch_size = self._states[0].size(1)
            new_state_batch_size = final_states[0].size(1)
            # Now we've sorted the states back so that they correspond to the original
            # indices, we need to figure out what states we need to update, because if we
            # didn't use a state for a particular row, we want to preserve its state.
            # Thankfully, the rows which are all zero in the state correspond exactly
            # to those which aren't used, so we create masks of shape (new_batch_size,),
            # denoting which states were used in the RNN computation.
            current_state_batch_size = self._states[0].size(1)
            new_state_batch_size = final_states[0].size(1)
            # Masks for the unused states of shape (1, new_batch_size, 1)
            used_new_rows_mask = [(state[0, :, :].sum(-1)
                                   != 0.0).float().view(1, new_state_batch_size, 1)
                                  for state in new_unsorted_states]
            new_states = []
            if current_state_batch_size > new_state_batch_size:
                # The new state is smaller than the old one,
                # so just update the indices which we used.
                for old_state, new_state, used_mask in zip(self._states,
                                                           new_unsorted_states,
                                                           used_new_rows_mask):
                    # zero out all rows in the previous state
                    # which _were_ used in the current state.
                    masked_old_state = old_state[:, :new_state_batch_size, :] * (1 - used_mask)
                    # The old state is larger, so update the relevant parts of it.
                    old_state[:, :new_state_batch_size, :] = new_state + masked_old_state
                    # Detatch the Variable.
                    new_states.append(torch.autograd.Variable(old_state.data))
            else:
                # The states are the same size, so we just have to
                # deal with the possibility that some rows weren't used.
                new_states = []
                for old_state, new_state, used_mask in zip(self._states,
                                                           new_unsorted_states,
                                                           used_new_rows_mask):
                    # zero out all rows which _were_ used in the current state.
                    masked_old_state = old_state * (1 - used_mask)
                    # The old state is larger, so update the relevant parts of it.
                    new_state += masked_old_state
                    # Detatch the Variable.
                    new_states.append(torch.autograd.Variable(new_state.data))

            # It looks like there should be another case handled here - when
            # the current_state_batch_size < new_state_batch_size. However,
            # this never happens, because the states themeselves are mutated
            # by appending zeros when calling _get_inital_states, meaning that
            # the new states are either of equal size, or smaller, in the case
            # that there are some unused elements (zero-length) for the RNN computation.
            self._states = tuple(new_states)

    def reset_states(self):
        self._states = None


class ElmobiLm(_EncoderBase):
    def __init__(self, config, use_cuda=False):
        super(ElmobiLm, self).__init__(stateful=True)
        self.config = config
        self.use_cuda = use_cuda
        input_size = config['encoder']['projection_dim']
        hidden_size = config['encoder']['projection_dim']
        cell_size = config['encoder']['dim']
        num_layers = config['encoder']['n_layers']
        memory_cell_clip_value = config['encoder']['cell_clip']
        state_projection_clip_value = config['encoder']['proj_clip']
        recurrent_dropout_probability = config['dropout']

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell_size = cell_size

        forward_layers = []
        backward_layers = []

        lstm_input_size = input_size
        go_forward = True
        for layer_index in range(num_layers):
            forward_layer = LstmCellWithProjection(lstm_input_size,
                                                  hidden_size,
                                                  cell_size,
                                                  go_forward,
                                                  recurrent_dropout_probability,
                                                  memory_cell_clip_value,
                                                  state_projection_clip_value)
            backward_layer = LstmCellWithProjection(lstm_input_size,
                                                   hidden_size,
                                                   cell_size,
                                                   not go_forward,
                                                   recurrent_dropout_probability,
                                                   memory_cell_clip_value,
                                                   state_projection_clip_value)
            if use_cuda:
                forward_layer = forward_layer.cuda()
                backward_layer = backward_layer.cuda()
            lstm_input_size = hidden_size

            self.add_module('forward_layer_{}'.format(layer_index), forward_layer)
            self.add_module('backward_layer_{}'.format(layer_index), backward_layer)
            forward_layers.append(forward_layer)
            backward_layers.append(backward_layer)
        self.forward_layers = forward_layers
        self.backward_layers = backward_layers

    def forward(self, inputs, mask):
        batch_size, total_sequence_length = mask.size()
        stacked_sequence_output, final_states, restoration_indices = \
            self.sort_and_run_forward(self._lstm_forward, inputs, mask)

        num_layers, num_valid, returned_timesteps, encoder_dim = stacked_sequence_output.size()
        # Add back invalid rows which were removed in the call to sort_and_run_forward.
        if num_valid < batch_size:
            zeros = stacked_sequence_output.data.new(num_layers,
                                                     batch_size - num_valid,
                                                     returned_timesteps,
                                                     encoder_dim).fill_(0)
            zeros = Variable(zeros)
            stacked_sequence_output = torch.cat([stacked_sequence_output, zeros], 1)

            # The states also need to have invalid rows added back.
            new_states = []
            for state in final_states:
                state_dim = state.size(-1)
                zeros = state.data.new(num_layers, batch_size - num_valid, state_dim).fill_(0)
                zeros = Variable(zeros)
                new_states.append(torch.cat([state, zeros], 1))
            final_states = new_states

        # It's possible to need to pass sequences which are padded to longer than the
        # max length of the sequence to a Seq2StackEncoder. However, packing and unpacking
        # the sequences mean that the returned tensor won't include these dimensions, because
        # the RNN did not need to process them. We add them back on in the form of zeros here.
        sequence_length_difference = total_sequence_length - returned_timesteps
        if sequence_length_difference > 0:
            zeros = stacked_sequence_output.data.new(num_layers,
                                                     batch_size,
                                                     sequence_length_difference,
                                                     stacked_sequence_output[0].size(-1)).fill_(0)
            zeros = Variable(zeros)
            stacked_sequence_output = torch.cat([stacked_sequence_output, zeros], 2)

        self._update_states(final_states, restoration_indices)

        # Restore the original indices and return the sequence.
        # Has shape (num_layers, batch_size, sequence_length, hidden_size)
        return stacked_sequence_output.index_select(1, restoration_indices)


    def _lstm_forward(self,
                      inputs: PackedSequence,
                      initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> \
        Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if initial_state is None:
            hidden_states: List[Optional[Tuple[torch.Tensor,
                                         torch.Tensor]]] = [None] * len(self.forward_layers)
        elif initial_state[0].size()[0] != len(self.forward_layers):
            raise Exception("Initial states were passed to forward() but the number of "
                            "initial states does not match the number of layers.")
        else:
            hidden_states = list(zip(initial_state[0].split(1, 0),
                                     initial_state[1].split(1, 0)))

        inputs, batch_lengths = pad_packed_sequence(inputs, batch_first=True)
        forward_output_sequence = inputs
        backward_output_sequence = inputs

        final_states = []
        sequence_outputs = []
        for layer_index, state in enumerate(hidden_states):
            forward_layer = getattr(self, 'forward_layer_{}'.format(layer_index))
            backward_layer = getattr(self, 'backward_layer_{}'.format(layer_index))

            forward_cache = forward_output_sequence
            backward_cache = backward_output_sequence

            # 이 부분에서 indentation 때문에 에러가 발생했었음!!
            if state is not None:
                forward_hidden_state, backward_hidden_state = state[0].split(self.hidden_size, 2)
                forward_memory_state, backward_memory_state = state[1].split(self.cell_size, 2)
                forward_state = (forward_hidden_state, forward_memory_state)
                backward_state = (backward_hidden_state, backward_memory_state)
            else:
                forward_state = None
                backward_state = None

            forward_output_sequence, forward_state = forward_layer(forward_output_sequence,
                                                                 batch_lengths,
                                                                 forward_state)
            backward_output_sequence, backward_state = backward_layer(backward_output_sequence,
                                                                    batch_lengths,
                                                                    backward_state)
            # Skip connections, just adding the input to the output.
            if layer_index != 0:
                forward_output_sequence += forward_cache
                backward_output_sequence += backward_cache

            sequence_outputs.append(torch.cat([forward_output_sequence,
                                               backward_output_sequence], -1))
            # Append the state tuples in a list, so that we can return
            # the final states for all the layers.
            final_states.append((torch.cat([forward_state[0], backward_state[0]], -1),
                                 torch.cat([forward_state[1], backward_state[1]], -1)))

        stacked_sequence_outputs: torch.FloatTensor = torch.stack(sequence_outputs)
        # Stack the hidden state and memory for each layer into 2 tensors of shape
        # (num_layers, batch_size, hidden_size) and (num_layers, batch_size, cell_size)
        # respectively.
        final_hidden_states, final_memory_states = zip(*final_states)
        final_state_tuple: Tuple[torch.FloatTensor,
                                 torch.FloatTensor] = (torch.cat(final_hidden_states, 0),
                                                       torch.cat(final_memory_states, 0))
        return stacked_sequence_outputs, final_state_tuple

class LstmCellWithProjection(torch.nn.Module):
    """
    An LSTM with Recurrent Dropout and a projected and clipped hidden state and
    memory. Note: this implementation is slower than the native Pytorch LSTM because
    it cannot make use of CUDNN optimizations for stacked RNNs due to and
    variational dropout and the custom nature of the cell state.
    Parameters
    ----------
    input_size : ``int``, required.
        The dimension of the inputs to the LSTM.
    hidden_size : ``int``, required.
        The dimension of the outputs of the LSTM.
    cell_size : ``int``, required.
        The dimension of the memory cell used for the LSTM.
    go_forward: ``bool``, optional (default = True)
        The direction in which the LSTM is applied to the sequence.
        Forwards by default, or backwards if False.
    recurrent_dropout_probability: ``float``, optional (default = 0.0)
        The dropout probability to be used in a dropout scheme as stated in
        `A Theoretically Grounded Application of Dropout in Recurrent Neural Networks
        <https://arxiv.org/abs/1512.05287>`_ . Implementation wise, this simply
        applies a fixed dropout mask per sequence to the recurrent connection of the
        LSTM.
    state_projection_clip_value: ``float``, optional, (default = None)
        The magnitude with which to clip the hidden_state after projecting it.
    memory_cell_clip_value: ``float``, optional, (default = None)
        The magnitude with which to clip the memory cell.
    Returns
    -------
    output_accumulator : ``torch.FloatTensor``
        The outputs of the LSTM for each timestep. A tensor of shape
        (batch_size, max_timesteps, hidden_size) where for a given batch
        element, all outputs past the sequence length for that batch are
        zero tensors.
    final_state: ``Tuple[torch.FloatTensor, torch.FloatTensor]``
        The final (state, memory) states of the LSTM, with shape
        (1, batch_size, hidden_size) and  (1, batch_size, cell_size)
        respectively. The first dimension is 1 in order to match the Pytorch
        API for returning stacked LSTM states.
    """
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 cell_size: int,
                 go_forward: bool = True,
                 recurrent_dropout_probability: float = 0.0,
                 memory_cell_clip_value: Optional[float] = None,
                 state_projection_clip_value: Optional[float] = None) -> None:
        super(LstmCellWithProjection, self).__init__()
        # Required to be wrapped with a :class:`PytorchSeq2SeqWrapper`.
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell_size = cell_size

        self.go_forward = go_forward
        self.state_projection_clip_value = state_projection_clip_value
        self.memory_cell_clip_value = memory_cell_clip_value
        self.recurrent_dropout_probability = recurrent_dropout_probability

        # We do the projections for all the gates all at once.
        self.input_linearity = nn.Linear(input_size, 4 * cell_size, bias=False)
        self.state_linearity = nn.Linear(hidden_size, 4 * cell_size, bias=True)

        # Additional projection matrix for making the hidden state smaller.
        self.state_projection = nn.Linear(cell_size, hidden_size, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        # Use sensible default initializations for parameters.
        block_orthogonal(self.input_linearity.weight.data, [self.cell_size, self.input_size])
        block_orthogonal(self.state_linearity.weight.data, [self.cell_size, self.hidden_size])

        self.state_linearity.bias.data.fill_(0.0)
        # Initialize forget gate biases to 1.0 as per An Empirical
        # Exploration of Recurrent Network Architectures, (Jozefowicz, 2015).
        self.state_linearity.bias.data[self.cell_size:2 * self.cell_size].fill_(1.0)

    def forward(self,  # pylint: disable=arguments-differ
                inputs: torch.FloatTensor,
                batch_lengths: List[int],
                initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """
        Parameters
        ----------
        inputs : ``torch.FloatTensor``, required.
            A tensor of shape (batch_size, num_timesteps, input_size)
            to apply the LSTM over.
        batch_lengths : ``List[int]``, required.
            A list of length batch_size containing the lengths of the sequences in batch.
        initial_state : ``Tuple[torch.Tensor, torch.Tensor]``, optional, (default = None)
            A tuple (state, memory) representing the initial hidden state and memory
            of the LSTM. The ``state`` has shape (1, batch_size, hidden_size) and the
            ``memory`` has shape (1, batch_size, cell_size).
        Returns
        -------
        output_accumulator : ``torch.FloatTensor``
            The outputs of the LSTM for each timestep. A tensor of shape
            (batch_size, max_timesteps, hidden_size) where for a given batch
            element, all outputs past the sequence length for that batch are
            zero tensors.
        final_state : ``Tuple[``torch.FloatTensor, torch.FloatTensor]``
            A tuple (state, memory) representing the initial hidden state and memory
            of the LSTM. The ``state`` has shape (1, batch_size, hidden_size) and the
            ``memory`` has shape (1, batch_size, cell_size).
        """
        batch_size = inputs.size()[0]
        total_timesteps = inputs.size()[1]

        # We have to use this '.data.new().fill_' pattern to create tensors with the correct
        # type - forward has no knowledge of whether these are torch.Tensors or torch.cuda.Tensors.
        output_accumulator = Variable(inputs.data.new(batch_size,
                                                      total_timesteps,
                                                      self.hidden_size).fill_(0))
        if initial_state is None:
            full_batch_previous_memory = Variable(inputs.data.new(batch_size,
                                                                  self.cell_size).fill_(0))
            full_batch_previous_state = Variable(inputs.data.new(batch_size,
                                                                 self.hidden_size).fill_(0))
        else:
            full_batch_previous_state = initial_state[0].squeeze(0)
            full_batch_previous_memory = initial_state[1].squeeze(0)

        current_length_index = batch_size - 1 if self.go_forward else 0
        if self.recurrent_dropout_probability > 0.0 and self.training:
            dropout_mask = get_dropout_mask(self.recurrent_dropout_probability,
                                            full_batch_previous_state)
        else:
            dropout_mask = None

        for timestep in range(total_timesteps):
            # The index depends on which end we start.
            index = timestep if self.go_forward else total_timesteps - timestep - 1

            # What we are doing here is finding the index into the batch dimension
            # which we need to use for this timestep, because the sequences have
            # variable length, so once the index is greater than the length of this
            # particular batch sequence, we no longer need to do the computation for
            # this sequence. The key thing to recognise here is that the batch inputs
            # must be _ordered_ by length from longest (first in batch) to shortest
            # (last) so initially, we are going forwards with every sequence and as we
            # pass the index at which the shortest elements of the batch finish,
            # we stop picking them up for the computation.
            if self.go_forward:
                while batch_lengths[current_length_index] <= index:
                    current_length_index -= 1
            # If we're going backwards, we are _picking up_ more indices.
            else:
                # First conditional: Are we already at the maximum number of elements in the batch?
                # Second conditional: Does the next shortest sequence beyond the current batch
                # index require computation use this timestep?
                while current_length_index < (len(batch_lengths) - 1) and \
                                batch_lengths[current_length_index + 1] > index:
                    current_length_index += 1

            # Actually get the slices of the batch which we
            # need for the computation at this timestep.
            # shape (batch_size, cell_size)
            previous_memory = full_batch_previous_memory[0: current_length_index + 1].clone()
            # Shape (batch_size, hidden_size)
            previous_state = full_batch_previous_state[0: current_length_index + 1].clone()
            # Shape (batch_size, input_size)
            timestep_input = inputs[0: current_length_index + 1, index]

            # Do the projections for all the gates all at once.
            # Both have shape (batch_size, 4 * cell_size)
            projected_input = self.input_linearity(timestep_input)
            projected_state = self.state_linearity(previous_state)

            # Main LSTM equations using relevant chunks of the big linear
            # projections of the hidden state and inputs.
            input_gate = torch.sigmoid(projected_input[:, (0 * self.cell_size):(1 * self.cell_size)] +
                                       projected_state[:, (0 * self.cell_size):(1 * self.cell_size)])
            forget_gate = torch.sigmoid(projected_input[:, (1 * self.cell_size):(2 * self.cell_size)] +
                                        projected_state[:, (1 * self.cell_size):(2 * self.cell_size)])
            memory_init = torch.tanh(projected_input[:, (2 * self.cell_size):(3 * self.cell_size)] +
                                     projected_state[:, (2 * self.cell_size):(3 * self.cell_size)])
            output_gate = torch.sigmoid(projected_input[:, (3 * self.cell_size):(4 * self.cell_size)] +
                                        projected_state[:, (3 * self.cell_size):(4 * self.cell_size)])
            memory = input_gate * memory_init + forget_gate * previous_memory

            # Here is the non-standard part of this LSTM cell; first, we clip the
            # memory cell, then we project the output of the timestep to a smaller size
            # and again clip it.

            if self.memory_cell_clip_value:
                # pylint: disable=invalid-unary-operand-type
                memory = torch.clamp(memory, -self.memory_cell_clip_value, self.memory_cell_clip_value)

            # shape (current_length_index, cell_size)
            pre_projection_timestep_output = output_gate * torch.tanh(memory)

            # shape (current_length_index, hidden_size)
            timestep_output = self.state_projection(pre_projection_timestep_output)
            if self.state_projection_clip_value:
                # pylint: disable=invalid-unary-operand-type
                timestep_output = torch.clamp(timestep_output,
                                              -self.state_projection_clip_value,
                                              self.state_projection_clip_value)

            # Only do dropout if the dropout prob is > 0.0 and we are in training mode.
            if dropout_mask is not None:
                timestep_output = timestep_output * dropout_mask[0: current_length_index + 1]

            # We've been doing computation with less than the full batch, so here we create a new
            # variable for the the whole batch at this timestep and insert the result for the
            # relevant elements of the batch into it.
            full_batch_previous_memory = Variable(full_batch_previous_memory.data.clone())
            full_batch_previous_state = Variable(full_batch_previous_state.data.clone())
            full_batch_previous_memory[0:current_length_index + 1] = memory
            full_batch_previous_state[0:current_length_index + 1] = timestep_output
            output_accumulator[0:current_length_index + 1, index] = timestep_output

        # Mimic the pytorch API by returning state in the following shape:
        # (num_layers * num_directions, batch_size, ...). As this
        # LSTM cell cannot be stacked, the first dimension here is just 1.
        final_state = (full_batch_previous_state.unsqueeze(0),
                       full_batch_previous_memory.unsqueeze(0))

        return output_accumulator, final_state
