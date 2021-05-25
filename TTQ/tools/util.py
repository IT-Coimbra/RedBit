import os
from shutil import copyfile
import argparse

def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

def clear(): #For Windows and Linux
    os.system('cls' if os.name=='nt' else 'clear')

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
