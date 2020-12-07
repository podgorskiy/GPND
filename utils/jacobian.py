import torch
from torch.autograd.gradcheck import zero_gradients
import numpy as np


def compute_jacobian_autograd(inputs, output):
    """
    :param inputs: Batch X Size (e.g. Depth X Width X Height)
    :param output: Batch X Classes
    :return: jacobian: Batch X Classes X Size
    """
    assert inputs.requires_grad

    num_classes = output.size()[1]

    jacobian = torch.zeros(num_classes, *inputs.size())
    grad_output = torch.zeros(*output.size())
    if inputs.is_cuda:
        grad_output = grad_output.cuda()
        jacobian = jacobian.cuda()

    for i in range(num_classes):
        zero_gradients(inputs)
        grad_output.zero_()
        grad_output[:, i] = 1
        output.backward(grad_output, retain_graph=True)
        jacobian[i] = inputs.grad.data

    return torch.transpose(jacobian, dim0=0, dim1=1)


def compute_jacobian_using_finite_differences(input, func, epsilon=1e-3):
    with torch.no_grad():
        r = func(input)
        output_size = np.prod(r.shape[1:]).item()
        input_size = np.prod(input.shape[1:]).item()
        J = torch.zeros(input_size, input.shape[0], output_size, requires_grad=False)
        J -= r.view(1, -1, output_size)

        onehot_shape = [-1] + list(input.shape[1:])

        for i in range(input_size):
            z_onehot = np.zeros([1, input_size], dtype=np.float32)
            z_onehot[0, i] = epsilon
            z_onehot = torch.tensor(z_onehot, dtype=torch.float32)
            d_recon_batch = func(input + z_onehot.reshape(onehot_shape))
            J[i] += d_recon_batch.view(-1, output_size)

        J /= epsilon

        J = torch.transpose(J, dim0=0, dim1=1)
        J = torch.transpose(J, dim0=1, dim1=2)

        return J


def compute_jacobian_using_finite_differences_v2(input, func, epsilon=1e-3):
    with torch.no_grad():
        input_size = np.prod(input.shape[1:]).item()
        input2 = torch.stack([input] * (input_size + 1)).view(input_size + 1, input.shape[0], input_size)

        for i in range(input_size):
            input2[i + 1, :, i] += epsilon

        y = func(input2.reshape([input.shape[0] * (input_size + 1)] + list(input.shape[1:])))

        output_size = np.prod(y.shape[1:]).item()

        J = torch.zeros(input_size, input.shape[0], output_size, requires_grad=False)
        J -= y[0].view(1, input.shape[0], output_size)
        J += y[1:].view(input_size, input.shape[0], output_size)
        J /= epsilon

        J = torch.transpose(J, dim0=0, dim1=1)
        J = torch.transpose(J, dim0=1, dim1=2)

        return J


def compute_jacobian_using_finite_differences_v3(input, func, epsilon=1e-3):
    with torch.no_grad():
        input_size = np.prod(input.shape[1:]).item()
        e = torch.eye(input_size, dtype=input.dtype).view(input_size, 1, input_size)
        input_ = input.view(1, input.shape[0], input_size)

        input2 = torch.stack([input_ + e * epsilon, input_ - e * epsilon])

        y = func(input2.reshape([input.shape[0] * input_size * 2] + list(input.shape[1:])))

        output_size = np.prod(y.shape[1:]).item()

        J = torch.zeros(input_size, input.shape[0], output_size, requires_grad=False)
        J += y[:input_size * input.shape[0]].view(input_size, input.shape[0], output_size)
        J -= y[input_size * input.shape[0]:].view(input_size, input.shape[0], output_size)
        J /= 2.0 * epsilon

        J = torch.transpose(J, dim0=0, dim1=1)
        J = torch.transpose(J, dim0=1, dim1=2)

        return J
