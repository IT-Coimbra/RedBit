import os
from shutil import copyfile
import argparse

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from collections import namedtuple

from tools import RunManager, get_num_correct, clear, str2bool, renameBestModel

from models import ResNet_cifar, VGG


# Training settings
parser = argparse.ArgumentParser(description='QNN PyTorch implementation (CIFAR-10 dataset)')

parser.add_argument('-n', '--network', default='ResNet',
					help='Neural network model to use: Choose ResNet or VGG')
parser.add_argument('-l', '--layers', type=int, default=20,
					help='Number of layers on the neural network model')

parser.add_argument('-o', '--optimizer', default='Adam',
					help='Optimizer to update weights')
parser.add_argument('-m', '--momentum', type=float, default=0.9,
					help='Momentum parameter for the SGD optimizer')

parser.add_argument('-wb', '--wbits', type=int, default=1,
					help='Bit width to represent weights')
parser.add_argument('-ab', '--abits', type=int, default=1,
					help='Bit width to represent input activations')

parser.add_argument('-bn', '--batch_norm', type=str2bool, nargs='?', const=True, default=True,
					help='If True, Batch Normalization is used (applicable for VGG; default=True)')

parser.add_argument('-lr', '--learning_rate', type=float, default=0.001,
					help='Initial Learning Rate')
parser.add_argument('-wd', '--weight_decay', type=float, default=1e-4,
					help='Weight decay parameter for optimizer')

parser.add_argument('-bs', '--batch_size', type=int, default=128,
					help='Batch Size for training and validation')
parser.add_argument('-e', '--epochs', type=int, default=200,
					help='Number of epochs to train')

parser.add_argument('-nw', '--number_workers', type=int, default=1,
					help='Number of workers on data loader')

parser.add_argument('-lc', '--load_checkpoint', type=str2bool, nargs='?', const=True, default=False,
					help='To resume training, set to True')


# Training function
def train(runManager):
	runManager.network.train()    #it tells pytorch we are in training mode
	
	runManager.begin_epoch()
	for images, targets in runManager.data_loader:
		images, targets = images.cuda(), targets.cuda()   #transfer to GPU

		preds = runManager.network(images)                 #Forward pass
		loss = runManager.criterion(preds, targets)           #Calculate loss

		# runManager.optimizer.zero_grad()
		for param in runManager.network.parameters():	# More efficient way to zero gradients
			param.grad = None
			
		loss.backward()
		for p in list(runManager.network.parameters()):
			if hasattr(p,'org'):
				p.data.copy_(p.org)
		runManager.optimizer.step()
		for p in list(runManager.network.parameters()):
			if hasattr(p,'org'):
				p.org.copy_(p.data)

		runManager.track_loss(loss)
		runManager.track_num_correct(preds, targets)
	runManager.end_epoch()

# Validation function
def val(runManager, best_acc, args):
	runManager.network.eval()   #it tells the model we are in validation mode

	val_correct = 0

	runManager.begin_epoch()
	for images, targets in runManager.data_loader:
		images, targets = images.cuda(), targets.cuda()

		preds = runManager.network(images)
		loss = runManager.criterion(preds, targets)
		
		runManager.track_loss(loss)
		runManager.track_num_correct(preds, targets)
		
		val_correct+=get_num_correct(preds, targets)
	
	acc=val_correct/len(runManager.data_loader.dataset)

	runManager.end_epoch()
	
	#Save checkpoint
	torch.save({
		'epoch': runManager.epoch.count,
		'accuracy': best_acc,
		'learning_rate': args.learning_rate,
		'network_state_dict': runManager.network.state_dict(),
		'optimizer_state_dict': runManager.optimizer.state_dict()
	}, f'trained_models/{args.networkCfg}.checkpoint.pth.tar')

	# Save best network model
	if acc >= best_acc:
		best_acc = acc
		copyfile(src=f'trained_models/{args.networkCfg}.checkpoint.pth.tar', 
				dst=f'trained_models/{args.networkCfg}.best.pth.tar')
	
	return best_acc

def adjust_learning_rate(optimizer, epoch, lr):
	"""Learning rate decays by 10 every 60 epochs until 180 epochs"""
	update_list = [60, 120, 180]
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
	
	# Parse arguments
	args = parser.parse_args()
	

	######################################
	##              Datasets            ##
	######################################
	train_set = datasets.CIFAR10(
		root= '../datasets/CIFAR10',
		train=True,
		download=True,
		transform=transforms.Compose([
			transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize((0.49139968, 0.48215841, 0.44653091), 
				(0.24703223, 0.24348513, 0.26158784))    
		])
	)
	val_set = datasets.CIFAR10(
		root= '../datasets/CIFAR10',
		train=False,
		download=True,
		transform=transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.49139968, 0.48215841, 0.44653091), 
				(0.24703223, 0.24348513, 0.26158784))    
		])
	)

	Params = namedtuple('Params',['lr','batch_size','number_workers'])
	params = Params( args.learning_rate, args.batch_size, args.number_workers)

	######################################
	##            Dataloaders           ##
	######################################
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
		network = ResNet_cifar(layers=args.layers, wbits=args.wbits, abits=args.abits).cuda()
	elif args.network.lower() == 'vgg' and args.layers in (11, 13, 16, 19):
		network = VGG(model_name[args.layers], wbits=args.wbits, abits=args.abits, batch_norm=args.batch_norm).cuda()
	else:
		print('Error creating network (network and/or configuration not supported)')
		exit()

	# Optimizer
	if args.optimizer.lower() == 'adam':
		optimizer = optim.Adam(network.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
	elif args.optimizer.lower() == 'sgd':
		optimizer = optim.SGD(network.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
	else:
		print('Invalid optimizer (use Adam or SGD)')
		exit()

	#Loss function
	criterion = torch.nn.CrossEntropyLoss().cuda()

	if args.batch_norm and args.network.lower() == 'vgg':
		bn = '.BN'
	else:
		bn = ''

	args.networkCfg = f'CIFAR-10.{args.network}-{args.layers}{bn}.QNN.W{args.wbits}.A{args.abits}.G32.{args.optimizer}.LR{args.learning_rate}'

	# Resume training
	if args.load_checkpoint:
		checkpoint = torch.load(f'trained_models/{args.networkCfg}.checkpoint.pth.tar')
		network.load_state_dict(checkpoint['network_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		start_epoch = checkpoint['epoch']	# Training will start in epoch+1
		best_acc = checkpoint['accuracy']
		lr = checkpoint['learning_rate']
	else:
		lr=args.learning_rate
		start_epoch=0       # Training will start in 0+1
		best_acc = 0.0

	trainManager = RunManager(f'{args.networkCfg}.Train', 'Train')
	validationManager = RunManager(f'{args.networkCfg}.Validation', 'Validation')

	trainManager.begin_run(params, network, train_loader, criterion, optimizer, start_epoch)
	validationManager.begin_run(params, network, val_loader, criterion, optimizer, start_epoch)
	for epoch in range(start_epoch, args.epochs):
		lr=adjust_learning_rate(optimizer, epoch, lr)
		trainManager.lr=lr
		validationManager.lr=lr
		args.learning_rate = lr
		train(trainManager)
		best_acc = val(validationManager, best_acc, args)
		clear()
		trainManager.printDF()
		validationManager.printDF()
		print(f'Best accuracy: {best_acc*100:.2f}%')
		
	trainManager.end_run()
	validationManager.end_run()
	renameBestModel(args, best_acc)
