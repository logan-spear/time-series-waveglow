import pandas as pd
import numpy as np

class DataLoader:
	def __init__(self, train_f="wind_power_data/wind_power_train.pickle", test_f = "wind_power_data/wind_power_test.pickle", n=96, rolling=True, small_subset=False, directory='.', valid_split=.2, use_gpu=False):
		print(directory+'/'+train_f)
		self.trainset = pd.read_pickle(directory+'/'+train_f).values
		self.testset = pd.read_pickle(directory+'/'+test_f).values
		print(len(self.testset))
		self.m = self.trainset.shape[0]
		self.m_test = self.testset.shape[0]
		self.n = n
		self.rolling = rolling
		self.small_subset = small_subset
		self.train_end = int(self.m*(1-valid_split))
		if self.rolling:
			if small_subset:
				print(self.n)
				print(self.m-self.n+1)
				self.sample_indices = np.random.choice(list(range(self.n, self.m-self.n+1)), 500, replace=False)
				self.valid_indices = np.random.choice(list(range(self.n, self.m-self.n+1)), 100, replace=False)
				# self.testset = np.random.choice(list(range(self.n, self.m-self.n+1)), 1000, replace=False)
				self.testset = self.testset[:2000]
				self.m_test = self.testset.shape[0]
			else:
				# self.sample_indices = np.random.choice(list(range(self.n, self.m-self.n+1)), self.m-2*self.n, replace=False)
				self.sample_indices = np.random.choice(list(range(self.n, self.train_end-self.n+1)), self.train_end-2*self.n, replace=False)
				self.valid_indices = list(range(self.train_end-self.n+1, self.m-self.n+1))
		else:
			self.sample_indices = np.random.choice(list(range(self.n, self.train_end-self.n+1, self.n)), int((self.train_end-self.n)/self.n), replace=False)
			self.valid_indices = list(range(self.train_end-self.n+1, self.m-self.n+1))


		self.num_samples = self.sample_indices.shape[0]
		self.sample_idx = 0
		self.epoch_end = True

	def sample(self, batch_size=24):
		if self.sample_idx+batch_size >= self.num_samples:
			self.epoch_end = False
			indices = self.sample_indices[self.sample_idx:]
			self.sample_idx = 0
			if self.rolling:
				if self.small_subset:
					self.sample_indices = np.random.choice(list(range(self.n, self.m-self.n+1)), 500, replace=False)
				else:
					self.sample_indices = np.random.choice(list(range(self.n, self.train_end-self.n+1)), self.train_end-2*self.n, replace=False)
			else:
				self.sample_indices = np.random.choice(list(range(self.n, self.train_end-self.n+1, self.n)), int((self.train_end-self.n)/self.n), replace=False)
		else:
			indices = self.sample_indices[self.sample_idx:self.sample_idx+batch_size]
			self.sample_idx += batch_size

		context = np.vstack([np.reshape(self.trainset[i-self.n:i], [1, self.n]) for i in indices])
		context = context[:, :, None]

		forecast = np.vstack([np.reshape(self.trainset[i:i+self.n], [1, self.n]) for i in indices])

		return context, forecast

	def valid_data(self):
		context = np.vstack([np.reshape(self.trainset[i-self.n:i], [1, self.n]) for i in self.valid_indices])
		forecast = np.vstack([np.reshape(self.trainset[i:i+self.n], [1, self.n]) for i in self.valid_indices])

		context = np.reshape(context, [len(self.valid_indices), self.n])
		context = context[:, :, None]
		forecast = np.reshape(forecast, [len(self.valid_indices), self.n])

		return context, forecast

	def test_samples(self, num_contexts=15):
		indices = np.random.choice(list(range(self.n, self.m_test-self.n+1, self.n)), num_contexts, replace=False)
		context = np.vstack([np.reshape(self.testset[i-self.n:i], [1, self.n]) for i in indices])
		forecast = np.vstack([np.reshape(self.testset[i:i+self.n], [1, self.n]) for i in indices])

		context = np.reshape(context, [num_contexts, self.n])
		context = context[:, :, None]
		forecast = np.reshape(forecast, [num_contexts, self.n])

		return context, forecast
	
	def test_data(self):
		context = np.vstack([np.reshape(self.testset[i:i+self.n], [1, self.n]) for i in range(self.testset.shape[0]-2*self.n)])
		context = context[:, :, None]
		forecast = np.vstack([np.reshape(self.testset[i:i+self.n], [1, self.n]) for i in range(self.n, self.testset.shape[0]-self.n)])
		return context, forecast
		

