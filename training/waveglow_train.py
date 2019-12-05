import torch
import numpy as np
import os
from waveglow_model import WaveGlow, WaveGlowLoss
import matplotlib.pyplot as plt
import argparse
from DataLoader import DataLoader
import pandas as pd
from train_utils import get_validation_loss, get_test_loss_and_mse, generate_tests
from utils import set_gpu_train_tensor, set_gpu_tensor, load_checkpoint, save_checkpoint

# n_context_channels=96, n_flows=6, n_group=24, n_early_every=3, n_early_size=8, n_layers=2, dilation_list=[1,2], n_channels=96, kernel_size=3, use_gpu=True
def training_procedure(dataset=None, num_gpus=0, output_directory='./train', epochs=1000, learning_rate=1e-4, batch_size=12, checkpointing=True, checkpoint_path="./checkpoints", seed=2019, params = [96, 6, 24, 3, 8, 2, [1,2], 96, 3], use_gpu=True, gen_tests=False, mname='model', validation_patience=10):
	params.append(use_gpu)
	torch.manual_seed(seed)
	if use_gpu:
		torch.cuda.manual_seed(seed)

#     if not os.path.isdir(output_directory[2:]): os.mkdir(output_directory[2:])
	if checkpointing and not os.path.isdir(checkpoint_path[2:]): os.mkdir(checkpoint_path[2:])
	criterion = WaveGlowLoss()
	model = WaveGlow(*params)
	if use_gpu:
		model.cuda()

	valid_context, valid_forecast = dataset.valid_data()
	valid_forecast = set_gpu_tensor(valid_forecast, use_gpu)
	valid_context = set_gpu_tensor(valid_context, use_gpu)

	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
	model.train()
	loss_iteration = []
	curr_validation = [np.inf]
	end_training = False
	for epoch in range(epochs):
		if end_training: break
		iteration = 0
		print("Epoch: %d/%d" % (epoch+1, epochs))
		avg_loss = []
		while(dataset.epoch_end):
			# model.zero_grad()
			context, forecast = dataset.sample(batch_size)
			forecast = set_gpu_train_tensor(forecast, use_gpu)
			context = set_gpu_train_tensor(context, use_gpu)
			z, log_s_list, log_det_w_list, early_out_shapes = model(forecast, context)

			loss = criterion((z, log_s_list, log_det_w_list))
			reduced_loss = loss.item()
			loss_iteration.append(reduced_loss)
			optimizer.zero_grad()
			loss.backward()
			avg_loss.append(reduced_loss)
			optimizer.step()
			print("Epoch [%d/%d] on iteration %d with loss %.4f" % (epoch+1, epochs, iteration, reduced_loss))
			iteration += 1

		epoch_loss = sum(avg_loss)/len(avg_loss)
		validation_loss = get_validation_loss(model, criterion, valid_context, valid_forecast)
		print("Epoch [%d/%d] had training loss: %.4f and validation_loss: %.4f" % (epoch+1, epochs, epoch_loss, validation_loss))
		
		if min(curr_validation) > validation_loss:
			print("Validation loss improved to %.5f" % validation_loss)
			curr_validation = [validation_loss]
			if gen_tests: generate_tests(dataset, model, 5, 96, use_gpu, str(epoch+1), mname=mname)
			if checkpointing:
				checkpoint_path = "%s/%s/epoch-%d_loss-%.4f" % (output_directory, mname, epoch, epoch_loss)
				save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path, use_gpu)
		else:
			curr_validation.append(validation_loss)
		dataset.epoch_end = True

		if len(curr_validation) == validation_patience: end_training = True

	if checkpointing:
		model, optimizer, iteration = load_checkpoint(checkpoint_path, model, optimizer)
		
	test_context, test_forecast = dataset.test_data()
	test_loss, test_mse = get_test_loss_and_mse(model, criterion, test_context, test_forecast, use_gpu, )

	if not checkpointing:
		checkpoint_path = "%s/%s/finalmodel_epoch-%d_testloss-%.4f_testmse_%.4f" % (output_directory, mname, epoch, test_loss, test_mse)
		save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path, use_gpu)
	
	print("Test loss for this model is %.5f, mse loss: %.5f" % (test_loss, test_mse))

	plt.figure()
	plt.plot(range(len(loss_iteration)), np.log10(np.array(loss_iteration)+1.0))
	plt.xlabel('iteration')
	plt.ylabel('log10 of loss')
	plt.savefig('%s/%s/total_loss_graph.png' % (output_directory, mname))
	plt.close()
	return test_loss, model


def run_training(config, epochs=100, batch_size=24, seed=2019, generate_per_epoch=False, generate_final_plots=True, checkpointing=True, use_gpu=False, n_channels=96, n_context_channels=96, rolling=True, datafile="wind_power_data/wind_power_train.pickle", valid_split=.2, small_subset=False, validation_patience=10):

	torch.manual_seed(1234)
	if use_gpu:
		torch.cuda.manual_seed(5678)

	dataset = DataLoader(train_f=datafile, rolling=rolling, small_subset=small_subset, valid_split=valid_split, use_gpu=use_gpu)
	params = [n_context_channels,
				config["n_flows"],
				config["n_group"],
				config["n_early_every"],
				config["n_early_size"],
				config["n_layers"],
				config["dilation_list"],
				n_channels,
				config["kernel_size"]]
	#     dataset = deepcopy(config["dataset"])
	# dataset = config["dataset"]
	#     if dataset==None:
	#         dataset = DataLoader(rolling=rolling, small_subset=False)
	output_directory = './train'

	if not os.path.isdir(output_directory):
		os.mkdir(output_directory)
		
	mname = 'waveglow_ncontextchannels-%d_nflows-%d_ngroup-%d-nearlyevery-%d-nearlysize-%d-nlayers-%d_dilations-%s_nchannels_%d-kernelsize-%d-lr-%.5f_seed-%d' % (params[0], params[1], params[2], params[3], params[4], params[5], str(params[6]), params[7], params[8], config["learning_rate"], seed)
	if not os.path.isdir(output_directory+"/"+mname):
		print("Making a new directory at " + output_directory+"/"+mname)
		os.mkdir(output_directory+'/'+mname)
	else:
		print("apparently directory already exists: " + output_directory +"/"+mname)
	   
	test_loss, final_model = training_procedure(epochs=epochs,    
								dataset=dataset, 
								use_gpu=use_gpu, 
								checkpointing=checkpointing, 
								gen_tests=generate_per_epoch, 
								batch_size=batch_size, 
								learning_rate=config["learning_rate"],
								seed=seed,
								params=params,
								mname=mname,
								output_directory=output_directory,
								validation_patience=validation_patience)


	print("Value of generate final plots: ", generate_final_plots)
	if generate_final_plots:
		print("Now calling generate_tests")
		generate_tests(dataset, final_model, use_gpu=use_gpu, mname=mname)

		print("Done with generate_tests function")

	

		
		
