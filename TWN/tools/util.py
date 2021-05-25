import torch
import torch.nn as nn
import numpy as np
import os
from shutil import copyfile
import argparse


def clear(): #For Windows and Linux
    os.system('cls' if os.name=='nt' else 'clear')

def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

class Ternary(torch.autograd.Function):

	@staticmethod
	def forward(self, input):
				 # **************** channel level-E(|W|) ****************
		if len(input.size()) == 4:
			E = torch.mean(torch.abs(input), (3, 2, 1), keepdim=True)
		elif len(input.size()) == 2:
			E = torch.mean(torch.abs(input), (1), keepdim=True)
				 # **************** Threshold ****************
		delta = E * 0.7
		# ************** W —— +-1、0 **************
		output = torch.sign(torch.add(torch.sign(torch.add(input, delta)),torch.sign(torch.add(input, -delta))))
		return output, delta

	@staticmethod
	def backward(self, grad_output, grad_delta):
		#*******************ste*********************
		grad_input = grad_output.clone()
		return grad_input

class TernarizeOp:
	def __init__(self,model):
		count_targets = 0
		for m in model.modules():
			if isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear):
				count_targets += 1
		self.ternarize_range = np.linspace(0,count_targets-1,count_targets).astype('int').tolist()
		self.num_of_params = len(self.ternarize_range)
		self.saved_params = []
		self.target_modules = []
		for m in model.modules():
			if isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear):
				tmp = m.weight.data.clone()
				self.saved_params.append(tmp) #tensor
				self.target_modules.append(m.weight) #Parameter
	
	def SaveWeights(self):
		for index in range(self.num_of_params):
			self.saved_params[index].copy_(self.target_modules[index].data)

	def TernarizeWeights(self):
		for index in range(self.num_of_params):
			self.target_modules[index].data = self.Ternarize(self.target_modules[index].data)
	
	def Ternarize(self,tensor):
		output_fp = tensor.clone()
		#output, delta = self.ternary(tensor)
		output, delta = Ternary.apply(tensor)
		
		output_abs = torch.abs(output_fp)
		mask_le = output_abs.le(delta)
		mask_gt = output_abs.gt(delta)
		output_abs[mask_le] = 0
		output_abs_th = output_abs.clone()
		if len(output_abs_th.size()) == 4:
			output_abs_th_sum = torch.sum(output_abs_th, (3, 2, 1), keepdim=True)
			mask_gt_sum = torch.sum(mask_gt, (3, 2, 1), keepdim=True).float()
		elif len(output_abs_th.size()) == 2:
			output_abs_th_sum = torch.sum(output_abs_th, (1), keepdim=True)
			mask_gt_sum = torch.sum(mask_gt, (1), keepdim=True).float()
		alpha = output_abs_th_sum / mask_gt_sum # α (scale factor)
		# *************** W * α ****************
		output = output * alpha

		return output			
	
	def Ternarization(self):
		self.SaveWeights()
		self.TernarizeWeights()
	
	def Restore(self):
		for index in range(self.num_of_params):
			self.target_modules[index].data.copy_(self.saved_params[index])
	
def str2bool(v):
	if isinstance(v, bool):
	   return v
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

def accuracy(output, target, topk=(1,)):
	"""Computes the precision@k for the specified values of k"""
	maxk = max(topk)
	batch_size = target.size(0)

	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)		# after upgrading to PyTorch with CUDA 10.2
		#correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
		res.append(correct_k.mul_(100.0 / batch_size))
	return res

def renameBestModel(args, accuracy):
	copyfile(src=f'trained_models/{args.networkCfg}.best.pth.tar', 
			dst=f'trained_models/{args.networkCfg}.Acc{accuracy*100:.2f}.pth.tar')

def renameBestModel_i(args, accuracy):
	copyfile(src=f'trained_models/{args.networkCfg}.best.pth.tar', 
			dst=f'trained_models/{args.networkCfg}.Acc{accuracy[0]:.2f}.top-1.{accuracy[1]:.2f}.top-5.pth.tar')