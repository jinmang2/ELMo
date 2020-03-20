import torch.nn as nn
from .utils import torch_variable
from .char_cnn import ConvTokenEmbedder

class Model(nn.Module):
    def __init__(self,
                 config,
                 word_emb_layer,
                 char_emb_layer,
                 n_class,
                 is_training=True,
                 use_cuda=False):
        super(Model, self).__init__()
        self.use_cuda = use_cuda
        self.config = config
        self.is_training = False

        self.token_embedder = ConvTokenEmbedder(
            config, word_emb_layer, char_emb_layer, use_cuda)
        self.encoder = ElmobiLM(config, use_cuda)
        self.output_dim = config['encoder']['projection_dim']
        self.classifier_name = config['classifier']['name'].lower()
        if self.classifier_name == 'softmax':
            self.classify_layer = SoftmaxLayer(self.output_dim, n_class)
        elif self.classifier_name == 'cnn_softmax':
            self.classify_layer = CNNSoftmaxLayer(self.token_embedder,
                                                  self.output_dim, n_class,
                                                  config['classifier']['n_samples'],
                                                  config['classifier']['corr_dim'],
                                                  use_cuda)
        elif self.classifier_name == 'sampled_softmax':
            self.classify_layer = SampledSoftmaxLayer(self.output_dim,
                                                      n_class,
                                                      config['classifier']['n_samples'],
                                                      use_cuda)
        else:
            raise Exception('')

    def forward(self, word_inp, chars_inp, mask_package):
        if (self.training and
            self.classifier_name in ['cnn_softmax', 'sampled_softmax']):
            self.classify_layer.update_negative_samples(word_inp, chars_inp, mask_package[0])
            self.classify_layer.update_embedding_matrix()

        token_embedding = self.token_embedder(word_inp, chars_inp, mask_package[0].size())
        token_embedding = F.dropout(token_embedding, self.config['dropout'], self.training)

        mask = torch_variable(mask_package[0], self.use_cuda)
        encoder_output = self.encoder(token_embedding, mask)
        encoder_output = encoder_output[1]
        encoder_output = F.dropout(encoder_output, self.config['dropout'], self.training)
        forward, backward = encoder_output.split(self.output_dim, 2)

        word_inp = torch_variable(word_inp, self.use_cuda)

        mask1 = torch_variable(mask_package[1], self.use_cuda)
        mask2 = torch_variable(mask_package[2], self.use_cuda)

        forward_x = forward.contiguous().view(-1, self.output_dim).index_select(0, mask1)
        forward_y = word_inp.contiguous().view(-1).view_select(0, mask2)

        backward_x = backward.contiguous().view(-1, self.output_dim).index_select(0, mask2)
        backward_y = word_inp.contiguous().view(-1).index_select(0, mask1)

        return (self.classify_layer(forward_x, forward_y),
                self.classify_layer(backward_x, backward_y))

    def save_model(self, path, save_classify_layer):
        torch.save(self.token_embedder.state_dict(), os.path.join(path, 'token_embedder.pkl'))
        torch.save(self.encoder.state_dict(), os.path.join(path, 'encoder.pkl'))
        if save_classify_layer:
            torch.save(self.classify_layer.state_dict(), os.path.join(path, 'classifier.pkl'))

    def load_model(self, path):
        self.token_embedder.load_state_dict(torch.load(os.path.join(path, 'token_embedder.pkl')))
        self.encoder.load_state_dict(torch.load(os.path.join(path, 'encoder.pkl')))
        self.classify_layer.load_state_dict(torch.load(os.path.join(path, 'classifier.pkl')))
