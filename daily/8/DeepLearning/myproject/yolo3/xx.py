from torch.autograd.function import Function
from torch._thnn import type2backend
import torch
class AdaptiveMaxPool2d(Function):
    def __init__(self, out_w, out_h):
        super(AdaptiveMaxPool2d, self).__init__()
        self.out_w = out_w
        self.out_h = out_h

    def forward(self, input):
        output = input.new()
        indices = input.new().long()
        self.save_for_backward(input)
        self.indices = indices
        self._backend = type2backend[input.type()]
        self._backend.SpatialAdaptiveMaxPooling_updateOutput(
            self._backend.library_state, input, output, indices,
            self.out_w, self.out_h)
        return output

    def backward(self, grad_output):
        input, = self.saved_tensors
        indices = self.indices
        grad_input = grad_output.new()
        self._backend.SpatialAdaptiveMaxPooling_updateGradInput(
            self._backend.library_state, input, grad_output, grad_input,
            indices)
        return grad_input, None

X=torch.arange(0,24)
X=X.float().view(1,4,6)
X.requires_grad_(True)
roi=AdaptiveMaxPool2d(3,3)
print(X)
Y=roi(X)
print(Y)
torch.sum(Y).backward()
print(X.grad)