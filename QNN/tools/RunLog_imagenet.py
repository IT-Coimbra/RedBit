from collections import OrderedDict
from collections import namedtuple
from itertools import product
from torch.utils.tensorboard import SummaryWriter
import torchvision

import time
import pandas as pd
from os import system, name


class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

class Epoch():
	def __init__(self):
		self.count = 0
		self.loss = AverageMeter()
		self.top1 = AverageMeter()
		self.top5 = AverageMeter()
		self.start_time = None

class Run():
	def __init__(self):
		self.data = []
		self.start_time = None
		self.duration = 0

class RunManager_i():
	def __init__(self, fileName, name):

		self.epoch = Epoch()
		self.run = Run()
		self.network = None
		self.data_loader = None
		self.criterion = None
		self.optimizer = None
		self.tb = None
		self.name = name
		self.fileName = fileName
		self.lr = 0

	def begin_run(self, params, network, loader, criterion, optimizer, start_epoch):
		self.run.start_time = time.time()
		self.network = network
		self.data_loader = loader
		self.criterion = criterion
		self.optimizer = optimizer
		self.tb = SummaryWriter(log_dir=f'runs/{self.fileName}-{params}')
		self.epoch.count = start_epoch
		if start_epoch == 0:
			self.tb.add_scalar(f'{self.name} Top-1 Accuracy', 0, self.epoch.count)
			self.tb.add_scalar(f'{self.name} Top-5 Accuracy', 0, self.epoch.count)

	def end_run(self):
		self.tb.close()
		self.epoch.count = 0
		self.run.duration = 0

	def begin_epoch(self):
		self.epoch.start_time = time.time()
		self.epoch.count += 1
		self.epoch.loss.reset()
		self.epoch.top1.reset()
		self.epoch.top5.reset()

	def end_epoch(self):
		epoch_duration = time.time() - self.epoch.start_time
		self.run.duration += epoch_duration

		self.tb.add_scalar(f'{self.name} Loss', self.epoch.loss.avg, self.epoch.count)
		self.tb.add_scalar(f'{self.name} Top-1 Accuracy', self.epoch.top1.avg, self.epoch.count)
		self.tb.add_scalar(f'{self.name} Top-5 Accuracy', self.epoch.top5.avg, self.epoch.count)

		# for name, weight in self.network.named_parameters():
		#     self.tb.add_histogram(name, weight, self.epoch.count)
		#     self.tb.add_histogram(f'{name}.grad', weight.grad, self.epoch.count)

		results = OrderedDict()
		results["Epoch"] = self.epoch.count
		results["Loss"] = self.epoch.loss.avg
		results["Top-1"] = self.epoch.top1.avg.item()
		results["Top-5"] = self.epoch.top5.avg.item()
		results["Epoch duration"] = epoch_duration
		results["Run duration"] = self.run.duration
		results["LR"] = self.lr

		self.run.data.append(results)

	def printDF(self):
		df = pd.DataFrame.from_dict(self.run.data, orient='columns')
		print(f'{self.name} table (Last 10 updates):')
		with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):      # Output all columns
			print(df.tail(10))


