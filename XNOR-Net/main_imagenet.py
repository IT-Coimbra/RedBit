import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import argparse
from collections import namedtuple
import torch.backends.cudnn as cudnn
import gc

import os
from shutil import copyfile

from models import ResNet_imagenet, AlexNet, VGG
from tools import RunManager_i, BinOp, clear, str2bool, accuracy, renameBestModel_i


def ParseArgs():
	parser = argparse.ArgumentParser(description='XNOR-Net Pytorch ImageNet Implementation.')
	parser.add_argument('-n', '--network', default='ResNet',
						help='Network model to use (ResNet or AlexNet or VGG)')

	parser.add_argument('-l', '--layers', type=int,
						help='Number of layers')

	parser.add_argument('-o', '--optimizer', default='Adam',
						help='Optimizer')

	parser.add_argument('-lr', '--learning_rate', type=float, default='1e-1',
						help='Initial Learning Rate')

	parser.add_argument('-wd', '--weight_decay', type=float, default='1e-4',
						help='Optimizer parameter weight decay')

	parser.add_argument('-m', '--momentum', type=float, default='0.9',
						help='SGD momentum parameter')

	parser.add_argument('-bs', '--batch_size', type=int, default='128',
						help='Batch Size')

	parser.add_argument('-e', '--epochs', type=int, default='100',
						help='Number of epochs to train')

	parser.add_argument('-nw', '--number_workers', type=int, default='4',
						help='Number of workers in DataLoader')

	parser.add_argument('-ba', '--binarize_activations', type=str2bool, nargs='?', const=True, default=True,
					help='If True, XNOR-Net will be performed, otherwise, BWN will be performed')

	parser.add_argument('-ir', '--insert_relu', type=str2bool, nargs='?', const=True, default=True,
					help='If True, ReLU will be introduced after a binary convolution layer')

	parser.add_argument('-bn', '--batch_norm', type=str2bool, nargs='?', const=True, default=True,
					help='If True, Batch Normalization is used (applicable for VGG; default=True)')

	parser.add_argument('-d', '--distributed', type=str2bool, nargs='?', const=True, default=True,
						help='If True, DataParallel will be used to train on multiple GPUs')

	parser.add_argument('-lc', '--load_checkpoint', type=str2bool, nargs='?', const=True, default=False,
						help='To resume training, set to True')
	
	args = parser.parse_args()
	return args
	

def train(runManager, bin_op, args):
	runManager.network.train()    #it tells pytorch we are in training mode

	runManager.begin_epoch()
	for images, targets in runManager.data_loader:
		images, targets = images.cuda(), targets.cuda()

		#binarize network
		bin_op.binarization()

		preds = runManager.network(images)                 #Forward pass
		loss=runManager.criterion(preds, targets)           #Calculate loss

		prec1, prec5 = accuracy(preds.data, targets, topk=(1, 5))
		runManager.epoch.loss.update(loss.data.item(), images.size(0))
		runManager.epoch.top1.update(prec1[0], images.size(0))
		runManager.epoch.top5.update(prec5[0], images.size(0))

		# runManager.optimizer.zero_grad()
		for param in runManager.network.parameters():	# More efficient way to zero gradients
			param.grad = None
			
		loss.backward()    #Calculate gradients

		#restore network to single precision
		bin_op.restore()
		bin_op.updateBinaryGradWeight()   #gradient for binary functions
			   
		runManager.optimizer.step()   #update weigths

		gc.collect()

	runManager.end_epoch()

def val(runManager, best_acc, bin_op, args):
	runManager.network.eval()   #it tells the model we are in validation mode

	#binarize network
	bin_op.binarization()

	runManager.begin_epoch()
	for images, targets in runManager.data_loader:
		images, targets = images.cuda(), targets.cuda()

		preds=runManager.network(images)
		loss=runManager.criterion(preds, targets)

		# measure accuracy and record loss
		prec1, prec5 = accuracy(preds.data, targets, topk=(1, 5))
		runManager.epoch.loss.update(loss.data.item(), images.size(0))
		runManager.epoch.top1.update(prec1[0], images.size(0))
		runManager.epoch.top5.update(prec5[0], images.size(0))

		gc.collect()

	runManager.end_epoch()
	
	#restore network to single precision
	bin_op.restore()
	
	#Save checkpoint
	if  torch.cuda.device_count() > 1 and args.distributed:
		torch.save({
			'epoch': runManager.epoch.count,
			'accuracy_top1': runManager.epoch.top1.avg,
			'accuracy_top5': runManager.epoch.top5.avg,
			'learning_rate': args.learning_rate,
			'network_state_dict': runManager.network.module.state_dict(),	# To generalize for loading later
			'optimizer_state_dict': runManager.optimizer.state_dict()
		}, f'trained_models/{args.networkCfg}.checkpoint.pth.tar')
	else:
		torch.save({
			'epoch': runManager.epoch.count,
			'accuracy_top1': runManager.epoch.top1.avg,
			'accuracy_top5': runManager.epoch.top5.avg,
			'learning_rate': args.learning_rate,
			'network_state_dict': runManager.network.state_dict(),
			'optimizer_state_dict': runManager.optimizer.state_dict()
		}, f'trained_models/{args.networkCfg}.checkpoint.pth.tar')

	# Save best network model
	if runManager.epoch.top1.avg >= best_acc[0]:
		best_acc[0] = runManager.epoch.top1.avg
		best_acc[1] = runManager.epoch.top5.avg
		copyfile(src=f'trained_models/{args.networkCfg}.checkpoint.pth.tar', 
				dst=f'trained_models/{args.networkCfg}.best.pth.tar')
	
	return best_acc
	
def adjust_learning_rate(optimizer, epoch, lr):
	"""Learning rate decays by 10 every 30 epochs until 90 epochs"""
	update_list = [30, 60, 90]
	new_lr=lr
	if epoch in update_list:
		new_lr = lr*0.1
		for param_group in optimizer.param_groups:
			param_group['lr'] = new_lr

	return new_lr

if __name__ == '__main__':
	if not os.path.exists('trained_models'):
		os.makedirs('trained_models')
	
	cudnn.benchmark = True

	args = ParseArgs()     

	train_set = datasets.ImageFolder(
		root= '../datasets/ImageNet/train',
		transform=transforms.Compose([
			transforms.Resize((256, 256)),
			transforms.RandomCrop(227),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406],
				std=[1./255., 1./255., 1./255.])    
		])
	)

	val_set = datasets.ImageFolder(
		root= '../datasets/ImageNet/val',
		transform=transforms.Compose([
		transforms.Resize((256, 256)),
		transforms.CenterCrop(227),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406],
			std=[1./255., 1./255., 1./255.])    
		])
	)

	Params = namedtuple('Params',['lr','batch_size','number_workers'])
	params = Params( args.learning_rate, args.batch_size, args.number_workers)

	train_loader = torch.utils.data.DataLoader(
		train_set,
		batch_size=args.batch_size,
		shuffle=True,
		num_workers=args.number_workers,
		pin_memory=True
	)
	val_loader = torch.utils.data.DataLoader(
		val_set,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=args.number_workers,
		pin_memory=True
	)
	
	model_name = {11: 'VGG11', 13: 'VGG13', 16: 'VGG16', 19: 'VGG19'}
	# Network model
	if args.network.lower() == 'resnet':
		network = ResNet_imagenet(layers=args.layers, binarize_activations=args.binarize_activations).cuda()
	elif args.network.lower() == 'vgg' and args.layers in (11, 13, 16, 19):
		network = VGG(model_name[args.layers], num_classes=1000, binarize_activations=args.binarize_activations, batch_norm=args.batch_norm).cuda()
	elif args.network.lower() == 'alexnet':
		network = AlexNet(binarize_activations=args.binarize_activations).cuda()
	else:
		print('Error creating network (network and or configuration not supported)')
		exit()

	if torch.cuda.device_count() > 1 and args.distributed:
		network = nn.DataParallel(network)

	criterion = nn.CrossEntropyLoss().cuda()

	if args.optimizer.lower() == 'sgd':
		optimizer = optim.SGD(network.parameters(),lr=args.learning_rate,momentum=args.momentum, weight_decay=args.weight_decay)
	elif args.optimizer.lower() == 'adam':
		optimizer = optim.Adam(network.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay)
		
	# define the binarization operator
	bin_op = BinOp(network)

	if args.batch_norm and args.network.lower() == 'vgg':
		bn = '.BN'
	else:
		bn = ''

	if args.insert_relu:
		relu='.ReLU'
	else:
		relu=''
	if args.binarize_activations:
		if args.network.lower() in ('resnet', 'vgg'):
			args.networkCfg = f'ImageNet.{args.network}-{args.layers}{bn}.XNOR-Net{relu}.{args.optimizer}.LR{args.learning_rate}'
		elif args.network.lower() == 'alexnet':
			args.networkCfg = f'ImageNet.{args.network}.XNOR-Net{relu}.{args.optimizer}.LR{args.learning_rate}'
	else:
		if args.network.lower() in ('resnet', 'vgg'):
			args.networkCfg = f'ImageNet.{args.network}-{args.layers}{bn}.BWN{relu}.{args.optimizer}.LR{args.learning_rate}'
		elif args.network.lower() == 'alexnet':
			args.networkCfg = f'ImageNet.{args.network}.BWN{relu}.{args.optimizer}.LR{args.learning_rate}'
	
	# Resume training
	if args.load_checkpoint:
		checkpoint = torch.load(f'trained_models/{args.networkCfg}.checkpoint.pth.tar')
		network.load_state_dict(checkpoint['network_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		start_epoch = checkpoint['epoch']	# Training will start in epoch+1
		best_acc = [0.0,0.0]
		best_acc[0] = checkpoint['accuracy_top1']
		best_acc[1] = checkpoint['accuracy_top1']
		lr = checkpoint['learning_rate']
	else:
		lr=args.learning_rate
		start_epoch=0       # Training will start in 0+1
		best_acc = [0.0,0.0]

	trainManager = RunManager_i(f'{args.networkCfg}.Train', 'Train')
	validationManager = RunManager_i(f'{args.networkCfg}.Validation', 'Validation')

	trainManager.begin_run(params, network, train_loader, criterion, optimizer, start_epoch)
	validationManager.begin_run(params, network, val_loader, criterion, optimizer, start_epoch)
	for epoch in range(start_epoch, args.epochs):
		lr = adjust_learning_rate(optimizer,epoch,lr)
		trainManager.lr=lr
		validationManager.lr=lr
		args.learning_rate = lr
		train(trainManager, bin_op, args)
		best_acc = val(validationManager, best_acc, bin_op, args)
		clear()
		trainManager.printDF()
		validationManager.printDF()
		print(f'Best accuracy: {best_acc[0]:.2f}% top-1; {best_acc[1]:.2f}% top-5')
		
	trainManager.end_run()
	validationManager.end_run()
	renameBestModel_i(args, best_acc)
