import torch
import torch.nn as nn
import math
from models.coordinate.linear import CoordinateLinear
from models.coordinate.utils import non_linear_symmetric, coordinate_function, coordinate_sigmoid

#f is typically tanh
class CoordinateLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, f=non_linear_symmetric(1)):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.f = coordinate_function(f)

        #Does this make sense?
        def reset_parameters_offset(offset):
            def reset_parameters_linear(self):
                min_max = 1.0 / math.sqrt(hidden_size)
                nn.init.uniform_(self.weight, -min_max, min_max)
                if self.bias is not None:
                    nn.init.uniform_(self.bias, -min_max + offset, min_max + offset)
            return reset_parameters_linear
        
        #TODO: handle bias better
        self.sigmoid_layer_1 = CoordinateLinear(input_size + hidden_size, hidden_size, bias=True,
                                                reset_parameters=reset_parameters_offset(1.))
        self.sigmoid_layer_2 = CoordinateLinear(input_size + hidden_size, hidden_size, bias=True,
                                                reset_parameters=reset_parameters_offset(0.))
        self.f_layer = CoordinateLinear(input_size + hidden_size, hidden_size, bias=bias,
                                        reset_parameters=reset_parameters_offset(0.))
        self.sigmoid_layer_3 = CoordinateLinear(input_size + hidden_size, hidden_size, bias=True,
                                                reset_parameters=reset_parameters_offset(0.))

    #http://colah.github.io/posts/2015-08-Understanding-LSTMs/ for reference
    def forward(self, inp, hidden, cell_state):
        assert len(inp) == self.input_size
        assert len(hidden) == self.hidden_size
        assert len(cell_state) == self.hidden_size
        total_in = inp + hidden

        sig_1 = coordinate_sigmoid(self.sigmoid_layer_1(total_in))
        sig_2 = coordinate_sigmoid(self.sigmoid_layer_2(total_in))
        f_l = self.f(self.f_layer(total_in))
        sig_3 = coordinate_sigmoid(self.sigmoid_layer_3(total_in))

        new_cell_state = []

        for i in range(self.hidden_size):
            new_cell_state.append(cell_state[i] * sig_1[i] + sig_2[i] * f_l[i])

        new_hidden_state = self.f(new_cell_state)

        for i in range(self.hidden_size):
            new_hidden_state[i] = new_hidden_state[i] * sig_3[i]

        return new_hidden_state, new_cell_state

if __name__ == '__main__':
    with torch.autograd.no_grad():
        lin = CoordinateLinear(10, 15)
        test_in = [torch.rand(1) for _ in range(10)]
        test_out = lin(test_in)
        true_out = torch.nn.functional.linear(torch.tensor(test_in), lin.weight, lin.bias)
        assert (torch.abs(torch.tensor(test_out) - true_out) < 1e-6).all()
