from torch.autograd import Variable

def torch_variable(tensor, use_cuda):
    tensor = Variable(tensor)
    if use_cuda:
        tensor = tensor.cuda()
    return tensor
