#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 20:08:30 2019

@author: lixiang
"""

import torch

class LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)
        return grad_input, grad_weight, grad_bias
    
class Linear(torch.nn.Module):
    def __init__(self, input_features, output_features, bias = None):
        super(Linear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.weight = torch.nn.Parameter(torch.Tensor(output_features, input_features))
        if bias is not None:
            self.bias = torch.nn.Parameter(torch.Tensor(output_features))
        else:
            self.register_parameter('bias', None)
        self.weight.data.uniform_(-0.1, 0.1)
        if bias is not None:
            self.bias.data.uniform_(-0.1, 0.1)
        self.linear = LinearFunction.apply
    def forward(self, input):
        return self.linear(input, self.weight, self.bias)
    def extra_repr(self):
        return 'input_features: {}, output_features: {}, bias: {}'.format(
                self.input_features, self.output_features, self.bias is not None
        )
        