import torch
import torch.nn as nn
import math

class CoordinateLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, reset_parameters=None):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        if reset_parameters is not None:
            self.reset_parameters = reset_parameters.__get__(self, CoordinateLinear)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inps):
        part_out = [[None for _ in range(self.in_features)] for _ in range(self.out_features)]

        assert len(inps) == self.in_features

        for i, inp in enumerate(inps):
            for j in range(self.out_features):
                part_out[j][i] = inp * self.weight[j][i]

        out = []
        for i, row in enumerate(part_out):
            if self.bias is not None:
                out.append(sum(row) + self.bias[i])
            else:
                out.append(sum(row))

        return out

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

if __name__ == '__main__':
    with torch.autograd.no_grad():
        lin = CoordinateLinear(10, 15)
        test_in = [torch.rand(1) for _ in range(10)]
        test_out = lin(test_in)
        true_out = torch.nn.functional.linear(torch.tensor(test_in), lin.weight, lin.bias)
        assert (torch.abs(torch.tensor(test_out) - true_out) < 1e-6).all()
