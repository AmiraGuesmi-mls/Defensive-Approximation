import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from approximate_multiplier import FP_appx_mul

class linear_appx(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias):
        ctx.save_for_backward(input, weight, bias)

        input = input.data.numpy()
        weight = weight.data.numpy()
        bias = bias.data.numpy()
        def appx_mul(A,B):
          window = np.zeros((A.shape[0],B.shape[1] ))
          for k in range(A.shape[0]):
            for l in range(B.shape[1]):
              for j in range(A.shape[1]):
                  window[k,l] +=  FP_appx_mul(A[k,j],B[j,l])
          return window

        #output = input.mm(weight.t()) + bias
        output = appx_mul(input,np.transpose(weight)) + bias
        return torch.from_numpy(output).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors

        grad_input = grad_output.mm(weight.float())
        grad_weight = grad_output.t().mm(input.float())
        grad_bias = grad_output.sum(0)
        return grad_input, grad_weight, grad_bias


class MyLinear(nn.Module):
    def __init__(self,in_features, out_features ):
        super(MyLinear, self).__init__()
        self.fn = linear_appx.apply
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        x = self.fn(x, self.weight, self.bias)
        return x