import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import argparse
from collections import namedtuple
import torch.backends.cudnn as cudnn

import os
from shutil import copyfile

from models import LeNet_5
from tools import RunManager, TernarizeOp, clear, get_num_correct, str2bool, renameBestModel


def ParseArgs():
	parser = argparse.ArgumentParser(description='Ternary-Weights-Network PyTorch MNIST.')
	parser.add_argument('-o', '--optimizer', default='SGD',
						help='Optimizer')

	parser.add_argument('-lr', '--learning_rate', type=float, default='1e-2',
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

	parser.add_argument('-t', '--ternarize', type=str2bool, nargs='?', const=True, default=True,
					help='If True, TWN will be performed, otherwise, FPWN will be performed')

	parser.add_argument('-lc', '--load_checkpoint', type=str2bool, nargs='?', const=True, default=False,
						help='To resume training, set to True')
	
	args = parser.parse_args()
	return args
	

def train(runManager, ternarize_op, args):
	runManager.network.train()    #it tells pytorch we are in training mode

	runManager.begin_epoch()
	for images, targets in runManager.data_loader:
		images, targets = images.cuda(), targets.cuda()

		if args.ternarize:
			ternarize_op.Ternarization()

		preds = runManager.network(images)                 #Forward pass
		loss=runManager.criterion(preds, targets)           #Calculate loss

		# runManager.optimizer.zero_grad()
		for param in runManager.network.parameters():	# More efficient way to zero gradients
			param.grad = None
			
		loss.backward()    #Calculate gradients

		if args.ternarize:
			ternarize_op.Restore()
			   
		runManager.optimizer.step()   #update weigths

		runManager.track_loss(loss)
		runManager.track_num_correct(preds, targets)
	runManager.end_epoch()

def val(runManager, best_acc, ternarize_op, args):
	runManager.network.eval()   #it tells the model we are in validation mode
	
	val_correct = 0

	if args.ternarize:
		ternarize_op.Ternarization()

	runManager.begin_epoch()
	for images, targets in runManager.data_loader:
		images, targets = images.cuda(), targets.cuda()

		preds=runManager.network(images)
		loss=runManager.criterion(preds, targets)
		
		runManager.track_loss(loss)
		runManager.track_num_correct(preds, targets)
		
		val_correct+=get_num_correct(preds, targets)
	acc=val_correct/len(runManager.data_loader.dataset)

	runManager.end_epoch()
	
	if args.ternarize:
		ternarize_op.Restore()
	
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
	"""Learning rate decays by 10 every 20 epochs until 80 epochs"""
	update_list = [20, 40, 60, 80]
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

	train_set = datasets.MNIST(
		root= '../datasets/',
		train=True,
		download=True,
		transform=transforms.Compose([
			transforms.RandomRotation(10),
			transforms.RandomCrop(28, padding=4),
			transforms.ToTensor(),
			transforms.Normalize((0.1307,), (0.3081,))
		])
	)

	val_set = datasets.MNIST(
		root= '../datasets/',
		train=False,
		download=True,
		transform=transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.1307,), (0.3081,))    
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

	network = LeNet_5().cuda()
	criterion = nn.CrossEntropyLoss().cuda()

	if args.optimizer.lower() == 'sgd':
		optimizer = optim.SGD(network.parameters(),lr=args.learning_rate,momentum=args.momentum, weight_decay=args.weight_decay)
	elif args.optimizer.lower() == 'adam':
		optimizer = optim.Adam(network.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay)
	
	if args.ternarize == True:
		ternarize_op = TernarizeOp(network)
		args.networkCfg = f'MNIST.LeNet-5.TWN.{args.optimizer}.LR{args.learning_rate}'
	else:
		ternarize_op = None
		args.networkCfg = f'MNIST.LeNet-5.FPWN.{args.optimizer}.LR{args.learning_rate}'
	
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
		lr = adjust_learning_rate(optimizer,epoch,lr)
		trainManager.lr=lr
		validationManager.lr=lr
		args.learning_rate = lr
		train(trainManager, ternarize_op, args)
		best_acc = val(validationManager, best_acc, ternarize_op, args)
		clear()
		trainManager.printDF()
		validationManager.printDF()
		print(f'Best accuracy: {best_acc*100:.2f}%')
		
	trainManager.end_run()
	validationManager.end_run()
	renameBestModel(args, best_acc)
