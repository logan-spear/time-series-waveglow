import numpy as np
import torch, os
import pandas as pd

def set_gpu_train_tensor(x, use_gpu):
	if use_gpu:
		x = torch.autograd.Variable(torch.cuda.FloatTensor(x))
	else:
		x = torch.autograd.Variable(torch.FloatTensor(x))
	return x

def set_gpu_tensor(x, use_gpu):
	if use_gpu:
		x = torch.cuda.FloatTensor(x).to('cuda')
	else:
		x = torch.FloatTensor(x)

	return x

def load_checkpoint(checkpoint_path, model, optimizer):
	assert(os.path.isfile(checkpoint_path))
	checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
	iteration = checkpoint_dict['iteration']
	optimizer.load_state_dict(checkpoint_dict['optimizer'])
	model_for_loading = checkpoint_dict['model']
	model.load_state_dict(model_for_loading.state_dict())
	print("Loaded checkpoint '%s' (iteration %d)" % (checkpoint_path, iteration))
	return model, optimizer, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, filepath, use_gpu=True):
	print("Saving model and optimizer state at iteration %d to %s" % (iteration, filepath))
	model_for_saving = model
	model_for_saving.load_state_dict(model.state_dict())
	torch.save({'model': model_for_saving,
				'iteration': iteration,
				'optimizer': optimizer.state_dict(),
				'learning_rate': learning_rate}, filepath)