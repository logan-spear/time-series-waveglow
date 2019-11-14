import torch
import numpy as np
import os
from model import WaveGlow
from utils import WaveGlowLoss


def load_checkpoint(checkpoint_path, model, optimizer):
	assert os.path.isfile(checkpoint_path)
	checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
	iteration = checkpoint_dict['iteration']
	optimizer.load_state_dict(checkpoint_dict['optimizer'])
	model_for_loading = checkpoint_dict['model']
    model.load_state_dict(model_for_loading.state_dict())
    print("Loaded checkpoint '%s' (iteration %d)" % (checkpoint_path, iteration))
    return model, optimizer, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, filepath):
	print("Saving model and optimizer state at iteration %d to %s" % (iteration, filepath))

	model_for_saving = WaveGlow().cuda()
	model_for_saving.load_state_dict(model.state_dict())
	torch.save({'model': model_for_saving,
				'iteration': iteration,
				'optimizer': optimizer.state_dict(),
				'learning_rate': learning_rate}, filepath)



def train(num_gpus, output_directory, epochs, learning_rate, batch_size, params, checkpointing=False, checkpoint_path="", seed=2019):
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)

	criterion = WaveGlowLoss()
	model = WaveGlow(params).cuda()

	optimizer = torch.optim.Adam(mode.parameters(), lr=learning_rate)

	iteration = 0
	if checkpoint_path != "":
		model, optimizer, iteration = load_checkpoint(checkpoint_path, model, optimizer)

		iteration += 1

	trainset = WindSamp()
	train_loader = DataLoader(trainset, num_workers=1, shuffle=False, 
								sampler=train_sampler, 
								batch_size=batch_size,
								pin_memory=False, 
								drop_last=True)


	model.train()
	for epoch in range(epochs):
		print("Epoch: %d/%d" % (epoch+1, epochs))
		for i, batch in enumerate(train_loader):

			model.zero_grad()
			context, forecast = batch

			context = torch.autograd.Variable(context.cuda())
			forecast = torch.autograd.Varible(forecast.cuda())
			outputs = model((context, forecast))

			loss = criterion(outputs)
			reduced_loss = loss.item()

			loss.backward()

			optimizer.step()
			print("On iteration %d with loss %.4f" % (iteration, reduced_loss))
			if (checkpointing and (iteration % iters_per_checkpoint == 0)):
				checkpoint_path = "%s/waveglow_%d_%.4f" % (output_directory)

				save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path)

			iteration += 1






if __name__ == "__main__":
	train()