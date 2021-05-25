import torch

def initial_scales(kernel):
	return 1.0, 1.0

def quantize(kernel, w_p, w_n, t):
	"""
	Return quantized weights of a layer.
	Only possible values of quantized weights are: {zero, w_p, -w_n}.
	"""
	delta = t*kernel.abs().max()
	a = (kernel > delta).float()
	b = (kernel < -delta).float()
	return w_p*a + (-w_n*b)


def get_grads(kernel_grad, kernel, w_p, w_n, t):
	"""
	Arguments:
		kernel_grad: gradient with respect to quantized kernel.
		kernel: corresponding full precision kernel.
		w_p, w_n: scaling factors.
		t: hyperparameter for quantization.

	Returns:
		1. gradient for the full precision kernel.
		2. gradient for w_p.
		3. gradient for w_n.
	"""
	delta = t*kernel.abs().max()
	# masks
	a = (kernel > delta).float()
	b = (kernel < -delta).float()
	c = torch.ones(kernel.size()).cuda() - a - b
	# scaled kernel grad and grads for scaling factors (w_p, w_n)
	return w_p*a*kernel_grad + w_n*b*kernel_grad + 1.0*c*kernel_grad,\
		(a*kernel_grad).sum(), (b*kernel_grad).sum()

