import numpy as np
import pandas as pd

class DataLoader:
	def __init__(self, train_f="wind_power_data/wind_power_train.pickle", test_f = "wind_power_data/wind_power_test.pickle", n=96, rolling=True):
		self.trainset = pd.read_pickle(train_f).values
		self.testset = pd.read_pickle(test_f).values
		self.m = self.trainset.shape[0]
		self.n = n
		if rolling:
			self.sample_indices = np.random.choice(list(range(self.n, self.m-self.n+1)), self.m-2*self.n, replace=False)
		else:
			self.sample_indices = np.random.choice(list(range(self.n, self.m-self.n+1, self.n), self.m-2*self.n, replace=False))
		self.num_samples = self.sample_indices.shape[0]
		self.sample_idx = 0
		self.epoch_end = True

	def sample(self, batch_size=24):

		if self.sample_idx+batch_size >= self.num_samples:
			self.epoch_end = False
			indices = self.sample_indices[self.sample_idx:]
			self.sample_idx = 0
			if rolling:
				self.sample_indices = np.random.choice(list(range(self.n, self.m-self.n+1)), self.m-2*self.n, replace=False)
			else:
				self.sample_indices = np.random.choice(list(range(self.n, self.m-self.n+1, self.n), self.m-2*self.n, replace=False))
		else:
			indices = self.sample_indices[self.sample_idx:self.sample_idx+batch_size]
			self.sample_idx += batch_size

		context = np.vstack([np.reshape(self.trainset[i-self.n:i], [1, self.n]) for i in indices])
		context = context[:, :, None]

		forecast = np.vstack([np.reshape(self.trainset[i:i+self.n], [1, self.n]) for i in indices])

		return context, forecast

