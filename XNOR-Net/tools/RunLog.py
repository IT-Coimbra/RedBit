from collections import OrderedDict
from collections import namedtuple
from itertools import product
from torch.utils.tensorboard import SummaryWriter
import torchvision

import time
import pandas as pd
from os import system, name


class Epoch():
	def __init__(self):
		self.count = 0
		self.loss = 0
		self.num_correct = 0
		self.start_time = None

class Run():
	def __init__(self):
		self.data = []
		self.start_time = None
		self.duration = 0

class RunManager():
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
			self.tb.add_scalar(f'{self.name} Accuracy', 0, self.epoch.count) #to start graph at zero

	def end_run(self):
		self.tb.close()
		self.epoch.count = 0
		self.run.duration = 0

	def begin_epoch(self):
		self.epoch.start_time = time.time()
		self.epoch.count += 1
		self.epoch.loss = 0
		self.epoch.num_correct = 0

	def end_epoch(self):
		epoch_duration = time.time() - self.epoch.start_time
		self.run.duration += epoch_duration
		loss = self.epoch.loss / len(self.data_loader.dataset)
		accuracy = self.epoch.num_correct / len(self.data_loader.dataset)

		self.tb.add_scalar(f'{self.name} Loss', loss, self.epoch.count)
		self.tb.add_scalar(f'{self.name} Accuracy', accuracy, self.epoch.count)

		for name, weight in self.network.named_parameters():
			self.tb.add_histogram(name, weight, self.epoch.count)
			self.tb.add_histogram(f'{name}.grad', weight.grad, self.epoch.count)

		results = OrderedDict()
		results["Epoch"] = self.epoch.count
		results["Loss"] = loss
		results["Accuracy"] = accuracy
		results["Epoch duration"] = epoch_duration
		results["Run duration"] = self.run.duration
		results["LR"] = self.lr
		
		self.run.data.append(results)
		
	def printDF(self):
		df = pd.DataFrame.from_dict(self.run.data, orient='columns')
		print(f'{self.name} table (Last 10 updates):')
		with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):      # Output all columns
			print(df.tail(10))

	def track_loss(self, loss):
		self.epoch.loss += loss.item() * self.data_loader.batch_size

	def track_num_correct(self, preds, targets):
		self.epoch.num_correct += self._get_num_correct(preds, targets)

	def _get_num_correct(self, preds, targets):
		return preds.argmax(dim=1).eq(targets).sum().item()
