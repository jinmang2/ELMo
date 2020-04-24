# -*- coding: utf-8 -*-

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# huggingface BERT에서 참고
class SpecialTokenMixin:

    TOKENS_ATTRIBUTES = [
        'bos_token',
        'eos_token',
        'bow_token',
        'eow_token',
        'unk_token',
        'sep_token',
        'pad_token',
        'cls_token',
        'mask_token',
        'oov_token',
        'additional_special_tokens',
    ]

    def __init__(self, **kwargs):
        self._bos_token  = None
        self._eos_token  = None
        self._bow_token  = None
        self._eow_token  = None
        self._unk_token  = None
        self._sep_token  = None
        self._pad_token  = None
        self._cls_token  = None
        self._mask_token = None
        self._oov_token  = None
        self._pad_token_type_id = 0
        self._additional_special_tokens = []

        for key, value in kwargs.items():
            if key in self.TOKENS_ATTRIBUTES:
                if key == 'additional_special_tokens':
                    assert isinstance(value, (list, tuple)) and \
                           all(isinstance(t, str) for t in value)
                # elif isinstance(vlue, AddedTokenFast):
                #     setattr(self, key, str(value))
                elif isinstance(value, str):
                    setattr(self, key, value)
                else:
                    raise ValueError(
                        f"special token {key} has to be either str "
                        f"or AddedTokenFast but got: {type(value)}"
                    )

    @property
    def bos_token(self):
        if self._bos_token is None:
            logger.error("Using bos_token, but it is not set yet.")
        return self._bos_token

    @property
    def eos_token(self):
        if self._eos_token is None:
            logger.error("Using eos_token, but it is not set yet.")
        return self._eos_token

    @property
    def bow_token(self):
        if self._bow_token is None:
            logger.error("Using bow_token, but it is not set yet.")
        return self._bow_token

    @property
    def eow_token(self):
        if self._eow_token is None:
            logger.error("Using eow_token, but it is not set yet.")
        return self._eow_token

    @property
    def unk_token(self):
        if self._unk_token is None:
            logger.error("Using unk_token, but it is not set yet.")
        return self._unk_token

    @property
    def sep_token(self):
        if self._sep_token is None:
            logger.error("Using sep_token, but it is not set yet.")
        return self._sep_token

    @property
    def pad_token(self):
        if self._pad_token is None:
            logger.error("Using pad_token, but it is not set yet.")
        return self._pad_token

    @property
    def cls_token(self):
        if self._cls_token is None:
            logger.error("Using cls_token, but it is not set yet.")
        return self._cls_token

    @property
    def mask_token(self):
        if self._mask_token is None:
            logger.error("Using mask_token, but it is not set yet.")
        return self._mask_token

    @property
    def oov_token(self):
        if self._oov_token is None:
            logger.error("Using oov_token, but it is not set yet.")
        return self._oov_token

    @property
    def additional_special_tokens(self):
        if self._additional_special_tokens is None:
            logger.error("Using additional_special_tokens, but it is not set yet.")
        return self.additional_special_tokens

    def _maybe_update_backend(self, value):
        """To be overriden by derived class if a backend tokenizer has to be updated."""
        pass

    @bos_token.setter
    def bos_token(self, value):
        self._bos_token = value
        self._maybe_update_backend([value])

    @eos_token.setter
    def eos_token(self, value):
        self._eos_token = value
        self._maybe_update_backend([value])

    @bow_token.setter
    def bow_token(self, value):
        self._bow_token = value
        self._maybe_update_backend([value])

    @eow_token.setter
    def eow_token(self, value):
        self._eow_token = value
        self._maybe_update_backend([value])

    @unk_token.setter
    def unk_token(self, value):
        self._unk_token = value
        self._maybe_update_backend([value])

    @sep_token.setter
    def sep_token(self, value):
        self._sep_token = value
        self._maybe_update_backend([value])

    @pad_token.setter
    def pad_token(self, value):
        self._pad_token = value
        self._maybe_update_backend([value])

    @cls_token.setter
    def cls_token(self, value):
        self._cls_token = value
        self._maybe_update_backend([value])

    @mask_token.setter
    def mask_token(self, value):
        self._mask_token = value
        self._maybe_update_backend([value])

    @oov_token.setter
    def oov_token(self, value):
        self._oov_token = value
        self._maybe_update_backend([value])

    @additional_special_tokens.setter
    def additional_special_tokens(self, value):
        self._additional_special_tokens = value
        self._maybe_update_backend([value])

    @property
    def bos_token_id(self):
        return self.convert_tokens_to_ids(self.bos_token)

    @property
    def eos_token_id(self):
        return self.convert_tokens_to_ids(self.eos_token)

    @property
    def bow_token_id(self):
        return self.convert_tokens_to_ids(self.bow_token)

    @property
    def eow_token_id(self):
        return self.convert_tokens_to_ids(self.eow_token)

    @property
    def unk_token_id(self):
        return self.convert_tokens_to_ids(self.unk_token)

    @property
    def sep_token_id(self):
        return self.convert_tokens_to_ids(self.sep_token)

    @property
    def pad_token_id(self):
        return self.convert_tokens_to_ids(self.pad_token)

    @property
    def cls_token_id(self):
        return self.convert_tokens_to_ids(self.cls_token)

    @property
    def mask_token_id(self):
        return self.convert_tokens_to_ids(self.mask_token)

    @property
    def oov_token_id(self):
        return self.convert_tokens_to_ids(self.oov_token)

    @property
    def additional_special_tokens_id(self):
        return self.convert_tokens_to_ids(self.additional_special_tokens)

    # @property
    # def special_tokens_map(self):
    #     """ A dictionary mapping special token class attribute (cls_token, unk_token...) to their
    #         values ('<unk>', '<cls>'...)
    #     """
    #     set_attr = {}
    #     for attr in self.SPECIAL_TOKENS_ATTRIBUTES:
    #         attr_value = getattr(self, "_" + attr)
    #         if attr_value:
    #             set_attr[attr] = attr_value
    #     return set_attr
    #
    # @property
    # def all_special_tokens(self):
    #     """ List all the special tokens ('<unk>', '<cls>'...) mapped to class attributes
    #         (cls_token, unk_token...).
    #     """
    #     all_toks = []
    #     set_attr = self.special_tokens_map
    #     print(set_attr)
    #     for attr_value in set_attr.values():
    #         all_toks = all_toks + (list(attr_value) if isinstance(attr_value, (list, tuple)) else [attr_value])
    #     all_toks = list(set(all_toks))
    #     return all_toks
    #
    # @property
    # def all_special_ids(self):
    #     """ List the vocabulary indices of the special tokens ('<unk>', '<cls>'...) mapped to
    #         class attributes (cls_token, unk_token...).
    #     """
    #     all_toks = self.all_special_tokens
    #     all_ids = self.convert_tokens_to_ids(all_toks)
    #     return all_ids

    def _convert_id_to_token(self, index):
        raise NotImplemented

    def _convert_token_to_id(self, token):
        raise NotImplemented

    def convert_tokens_to_ids(self, tokens):
        raise NotImplemented

    def convert_ids_to_tokens(self, ids):
        raise NotImplemented
