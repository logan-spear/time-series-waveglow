import torch
from DataLoader import DataLoader
import numpy as np
import os
from waveglow_model import WaveGlow, WaveGlowLoss
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='argument parser')
parser.add_argument('--epochs', dest='epochs', type=int, default=100)
parser.add_argument('--rolling', dest='rolling', type=int, default=1)
parser.add_argument('--small_subset', dest='small_subset', type=int, default=0)
parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1)
parser.add_argument('--checkpointing', dest='checkpointing', type=int, default=1)
parser.add_argument('--generate_per_epoch', dest='generate_per_epoch', type=int, default=1)
parser.add_argument('--generate_final', dest='generate_final', type=int, default=1)
args = parser.parse_args()

args.rolling = True if args.rolling else False
args.small_subset = True if args.small_subset else False
args.use_gpu = True if args.use_gpu else False
args.checkpointing = True if args.checkpointing else False

def load_checkpoint(checkpoint_path, model, optimizer):
	assert os.path.isfile(checkpoint_path)
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


# n_context_channels=96, n_flows=6, n_group=24, n_early_every=3, n_early_size=8, n_layers=2, dilation_list=[1,2], n_channels=96, kernel_size=3, use_gpu=True
def training(dataset=None, num_gpus=0, output_directory='./train', epochs=1000, learning_rate=1e-4, batch_size=12, checkpointing=True, checkpoint_path="./checkpoints", seed=2019, params = [96, 6, 24, 3, 8, 2, [1,2], 96, 3], use_gpu=True, gen_tests=True):
	print("#############")
	print(use_gpu)
	params.append(use_gpu)
	torch.manual_seed(seed)
	if use_gpu:
		torch.cuda.manual_seed(seed)

	if not os.path.isdir(output_directory[2:]): os.mkdir(output_directory[2:])
	if checkpointing and not os.path.isdir(checkpoint_path[2:]): os.mkdir(checkpoint_path[2:])
	criterion = WaveGlowLoss()
	model = WaveGlow(*params)
	if use_gpu:
		model.cuda()

	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

	# iteration = 0
	# if checkpoint_path != "":
		# model, optimizer, iteration = load_checkpoint(checkpoint_path, model, optimizer)

		# iteration += 1


	model.train()
	loss_iteration = []
	for epoch in range(epochs):
		iteration = 0
		print("Epoch: %d/%d" % (epoch+1, epochs))
		avg_loss = []
		while(dataset.epoch_end):
			model.zero_grad()
			context, forecast = dataset.sample(batch_size)

			if use_gpu:
				forecast = torch.autograd.Variable(torch.cuda.FloatTensor(forecast))
				context = torch.autograd.Variable(torch.cuda.FloatTensor(context))
			else:
				forecast = torch.autograd.Variable(torch.FloatTensor(forecast))
				context = torch.autograd.Variable(torch.FloatTensor(context))
			
			z, log_s_list, log_det_w_list, early_out_shapes = model(forecast, context)

			loss = criterion((z, log_s_list, log_det_w_list))
			reduced_loss = loss.item()
			loss_iteration.append(reduced_loss)
			loss.backward()
			avg_loss.append(reduced_loss)
			optimizer.step()
			print("On iteration %d with loss %.4f" % (iteration, reduced_loss))
			iteration += 1
			# if (checkpointing and (iteration % iters_per_checkpoint == 0)):

		if gen_tests: generate_tests(dataset, model, 5, 96, use_gpu, str(epoch+1))
		epoch_loss = sum(avg_loss)/len(avg_loss)
		if checkpointing:
			checkpoint_path = "%s/waveglow_epoch-%d_%.4f" % (output_directory, epoch, epoch_loss)
			save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path, use_gpu)


			
		dataset.epoch_end = True
		plt.figure()
		plt.plot(range(len(loss_iteration)), np.log10(loss_iteration))
		plt.xlabel('iteration')
		plt.ylabel('log10 of loss')
		plt.savefig('total_loss_graph.png')
		plt.close()
	return model

def generate_tests(dataset, model, num_contexts=15, n=96, use_gpu=True, epoch='final'):
	context, forecast = dataset.test_samples(num_contexts=15)

	if use_gpu:
		context = torch.cuda.FloatTensor(context)
	else:
		context = torch.FloatTensor(context)

	if use_gpu:
		gen_forecast = model.generate(context).cpu()
	else:
		gen_forecast = model.generate(context)

	for i in range(num_contexts):
		plt.figure()
		plt.plot(range(n), gen_forecast[i, :], label='generated')
		plt.plot(range(n), forecast[i, :], label='original')
		plt.legend()
		plt.xlabel('time (t)')
		plt.savefig('forecast_generated_%d_epoch-%s.png' % (i, epoch))
		plt.close()



if __name__ == "__main__":
	
	dataset = DataLoader(rolling=args.rolling, small_subset=args.small_subset)
	final_model = training(epochs=args.epochs, dataset=dataset, use_gpu=args.use_gpu, checkpointing=args.checkpointing, gen_tests=args.generate_per_epoch)
	if args.generate_final:
		generate_tests(dataset, final_model, use_gpu=args.use_gpu)

