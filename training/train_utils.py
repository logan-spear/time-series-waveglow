import numpy as np
import torch, os
from utils import set_gpu_tensor
import matplotlib.pyplot as plt
import pandas as pd

def generate_tests(dataset, model, num_contexts=15, n=96, use_gpu=True, epoch='final', batch_size=24, output_directory='generated', mname='model'):
	if not os.path.isdir(output_directory):
		os.mkdir(output_directory)
		
	context, forecast = dataset.test_samples(num_contexts=num_contexts)
	if use_gpu:
		context = torch.cuda.FloatTensor(context)
	else:
		context = torch.FloatTensor(context)

	if use_gpu:
		gen_forecast = model.generate(context).cpu()
	else:
		gen_forecast = model.generate(context)
		
	if not os.path.isdir(output_directory + "/" + mname):
		os.mkdir(output_directory+"/"+mname)
		
	print("Done generating forecasts using model, now generating plots")

	for i in range(num_contexts):
		plt.figure()
		plt.plot(range(n), gen_forecast[i, :], label='generated')
		plt.plot(range(n), forecast[i, :], label='original')
		plt.legend()
		plt.xlabel('time (t)')
		plt.savefig('%s/%s/forecast_generated_%d_epoch-%s.png' % (output_directory, mname, i, epoch))
		plt.close()
		print("Generated plot %i of %i" % (i+1, num_contexts))
	
	print("End of generate tests function")

def test_mse_loss(context, forecast, model, use_gpu, generations_per_sample=4):
	mse_loss = 0.0
	for i in range(generations_per_sample):
		if use_gpu:
			gen_forecast = model.generate(context)
		else:
			gen_forecast = model.generate(context).cpu()
		mse_loss += np.square(gen_forecast-forecast).mean(axis=1).mean()#np.square(gen_forecast-forecast).mean(axis=0)
		

	# print("Test MSE Loss: %.4f" % mse_loss)
	return mse_loss

def get_validation_loss(model, criterion, valid_context, valid_forecast):
	z, log_s_list, log_det_w_list, early_out_shapes = model(valid_forecast, valid_context)
	loss = criterion((z, log_s_list, log_det_w_list))
	return loss.item()

def get_validation_loss_and_mse(model, criterion, valid_context, valid_forecast, use_gpu):
	z, log_s_list, log_det_w_list, early_out_shapes = model(valid_forecast, valid_context)
	loss = criterion((z, log_s_list, log_det_w_list))
	test_mse = test_mse_loss(context_batch, forecast_batch, model, use_gpu)
	return loss.item()

def get_test_loss_and_mse(model, criterion, context, forecast, use_gpu):
	test_loss = 0; test_mse = 0; test_batch = 1000; batches = 0
	for i in range(0, context.shape[0], test_batch):
		batches+=1
		context_batch = set_gpu_tensor(context[i:i+(test_batch if i+test_batch < context.shape[0] else context.shape[0])], use_gpu)
		forecast_batch = set_gpu_tensor(forecast[i:i+(test_batch if i+test_batch < context.shape[0] else context.shape[0])], use_gpu)

		z, log_s_list, log_det_w_list, early_out_shapes = model(forecast_batch, context_batch)
		loss = criterion((z, log_s_list, log_det_w_list))
		test_loss += loss.item()
		test_mse += test_mse_loss(context_batch, forecast_batch, model, use_gpu)

	test_mse = test_mse/batches
	test_loss = test_loss/batches

	return test_loss, test_mse


def get_test_loss(model, criterion, context, forecast, use_gpu):
	test_loss = 0; test_batch = 10000; batches = 0
	for i in range(0, context.shape[0], test_batch):
		batches+=1
		context_batch = set_gpu_tensor(context[i:i+(test_batch if i+test_batch < context.shape[0] else context.shape[0])], use_gpu)
		forecast_batch = set_gpu_tensor(forecast[i:i+(test_batch if i+test_batch < context.shape[0] else context.shape[0])], use_gpu)

		z, log_s_list, log_det_w_list, early_out_shapes = model(forecast_batch, context_batch)
	
		loss = criterion((z, log_s_list, log_det_w_list))
		test_loss += loss.item()

	test_loss = test_loss/batches

	return test_loss, test_mse


