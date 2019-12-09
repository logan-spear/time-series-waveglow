import utils, waveglow_model
import DataLoader
from waveglow_model import WaveGlowLoss
from train_utils import get_test_loss_and_mse
import torch
import numpy as np


dataset = DataLoader.DataLoader(test_f="./wind_power_data/wind_power_test.pickle", train_f="./wind_power_data/wind_power_train.pickle", rolling=True, small_subset=False, valid_split=.2, use_gpu=False)
criterion = WaveGlowLoss()
test_context, test_forecast = dataset.test_data()

context1s = []; context2s = []; context3s = []; context4s = []; context5s = []; context6s = []; context7s = []; context8s = []; context9s = []

i = 96
context1 = torch.cuda.FloatTensor(test_context[i:i+96])
context2 = torch.cuda.FloatTensor(test_context[i+96:i+96*2])
context3 = torch.cuda.FloatTensor(test_context[i+96*2:i+96*3])
context4 = torch.cuda.FloatTensor(test_context[i+96*3:i+96*4])
context5 = torch.cuda.FloatTensor(test_context[i+96*4:i+96*5])
context6 = torch.cuda.FloatTensor(test_context[i+96*5:i+96*6])
context7 = torch.cuda.FloatTensor(test_context[i+96*6:i+96*7])
context8 = torch.cuda.FloatTensor(test_context[i+96*7:i+96*8])
context9 = torch.cuda.FloatTensor(test_context[i+96*8:i+96*9])
forecast1 = torch.cuda.FloatTensor(test_context[i+96:i+96+96])
forecast2 = torch.cuda.FloatTensor(test_context[i+96+96:i+96*2+96])
forecast3 = torch.cuda.FloatTensor(test_context[i+96*2+96:i+96*3+96])
forecast4 = torch.cuda.FloatTensor(test_context[i+96*3+96:i+96*4+96])
forecast5 = torch.cuda.FloatTensor(test_context[i+96*4+96:i+96*5+96])
forecast6 = torch.cuda.FloatTensor(test_context[i+96*5+96:i+96*6+96])
forecast7 = torch.cuda.FloatTensor(test_context[i+96*6+96:i+96*7+96])
forecast8 = torch.cuda.FloatTensor(test_context[i+96*7+96:i+96*8+96])
forecast9 = torch.cuda.FloatTensor(test_context[i+96*8+96:i+96*9+96])



model = waveglow_model.WaveGlow(
    n_context_channels=96, 
    n_flows=4, 
    n_group=12, 
    n_early_every=99,
    n_early_size=99,
    n_layers=8,
    dilation_list=[1,1,2,2,2,2,4,4],
    n_channels=96,
    kernel_size=3, use_cuda=True);

model, iteration_num = utils.load_checkpoint("./checkpoints/epoch-12_loss--0.3116", model)
model.cuda()
test_loss, test_mse = get_test_loss_and_mse(model, criterion, test_context, test_forecast, True)
print("Loss and MSE for model %s"% "waveglow_ncontextchannels-96_nflows-4_ngroup-12-nearlyevery-99-nearlysize-99-nlayers-8_dilations-1-1-2-2-2-2-4-4_nchannels_96-kernelsize-3-lr-0.00100_seed-2019")
print(test_loss)
print(test_mse)

for i in range(5):
	context1s.append(model.generate(context1).cpu())
	context2s.append(model.generate(context2).cpu())
	context3s.append(model.generate(context3).cpu())
	context4s.append(model.generate(context4).cpu())
	context5s.append(model.generate(context5).cpu())
	context6s.append(model.generate(context6).cpu())
	context7s.append(model.generate(context7).cpu())
	context8s.append(model.generate(context8).cpu())
	context9s.append(model.generate(context9).cpu())

# 5 models, 9 contexts, 5 generations per contex = 45 total


model = waveglow_model.WaveGlow(
    n_context_channels=96, 
    n_flows=4, 
    n_group=24, 
    n_early_every=99,
    n_early_size=99,
    n_layers=16,
    dilation_list=[1,1,2,2,2,2,2,2,2,2,2,2,2,2,4,4],
    n_channels=96,
    kernel_size=9, use_cuda=True);

model, iteration_num = utils.load_checkpoint("./checkpoints/epoch-19_loss--0.2139", model)
model.cuda()
test_loss, test_mse = get_test_loss_and_mse(model, criterion, test_context, test_forecast, True)
print("Loss and MSE for model %s"% "waveglow_ncontextchannels-96_nflows-4_ngroup-24-nearlyevery-99-nearlysize-99-nlayers-16_dilations-1-1-2-2-2-2-2-2-2-2-2-2-2-2-4-4_nchannels_96-kernelsize-9-lr-0.00100_seed-2019")
print(test_loss)
print(test_mse)

for i in range(5):
	context1s.append(model.generate(context1).cpu())
	context2s.append(model.generate(context2).cpu())
	context3s.append(model.generate(context3).cpu())
	context4s.append(model.generate(context4).cpu())
	context5s.append(model.generate(context5).cpu())
	context6s.append(model.generate(context6).cpu())
	context7s.append(model.generate(context7).cpu())
	context8s.append(model.generate(context8).cpu())
	context9s.append(model.generate(context9).cpu())


model = waveglow_model.WaveGlow(
    n_context_channels=96, 
    n_flows=4, 
    n_group=24, 
    n_early_every=99,
    n_early_size=99,
    n_layers=4,
    dilation_list=[1,1,2,2],
    n_channels=96,
    kernel_size=9, use_cuda=True);

model, iteration_num = utils.load_checkpoint("./checkpoints/epoch-22_loss--0.2237", model)
model.cuda()
test_loss, test_mse = get_test_loss_and_mse(model, criterion, test_context, test_forecast, True)
print("Loss and MSE for model %s"% "waveglow_ncontextchannels-96_nflows-4_ngroup-24-nearlyevery-99-nearlysize-99-nlayers-4_dilations-1-1-2-2_nchannels_96-kernelsize-9-lr-0.00100_seed-2019")
print(test_loss)
print(test_mse)

for i in range(5):
	context1s.append(model.generate(context1).cpu())
	context2s.append(model.generate(context2).cpu())
	context3s.append(model.generate(context3).cpu())
	context4s.append(model.generate(context4).cpu())
	context5s.append(model.generate(context5).cpu())
	context6s.append(model.generate(context6).cpu())
	context7s.append(model.generate(context7).cpu())
	context8s.append(model.generate(context8).cpu())
	context9s.append(model.generate(context9).cpu())


model = waveglow_model.WaveGlow(
    n_context_channels=96, 
    n_flows=16, 
    n_group=48, 
    n_early_every=99,
    n_early_size=99,
    n_layers=4,
    dilation_list=[1,1,2,2],
    n_channels=96,
    kernel_size=9, use_cuda=True);

model, iteration_num = utils.load_checkpoint("./checkpoints/epoch-36_loss—.1936", model)
model.cuda()
test_loss, test_mse = get_test_loss_and_mse(model, criterion, test_context, test_forecast, True)
print("Loss and MSE for model %s"% "waveglow_ncontextchannels-96_nflows-4_ngroup-24-nearlyevery-99-nearlysize-99-nlayers-4_dilations-1-1-2-2_nchannels_96-kernelsize-9-lr-0.00100_seed-2019")
print(test_loss)
print(test_mse)

for i in range(5):
	context1s.append(model.generate(context1).cpu())
	context2s.append(model.generate(context2).cpu())
	context3s.append(model.generate(context3).cpu())
	context4s.append(model.generate(context4).cpu())
	context5s.append(model.generate(context5).cpu())
	context6s.append(model.generate(context6).cpu())
	context7s.append(model.generate(context7).cpu())
	context8s.append(model.generate(context8).cpu())
	context9s.append(model.generate(context9).cpu())




model = waveglow_model.WaveGlow(
    n_context_channels=96, 
    n_flows=8, 
    n_group=24, 
    n_early_every=99,
    n_early_size=99,
    n_layers=8,
    dilation_list=[1,1,2,2,2,2,4,4],
    n_channels=96,
    kernel_size=5, use_cuda=True);

model, iteration_num = utils.load_checkpoint("./checkpoints/epoch_9_loss—.224", model)
model.cuda()
test_loss, test_mse = get_test_loss_and_mse(model, criterion, test_context, test_forecast, True)
print("Loss and MSE for model %s"% "waveglow_ncontextchannels-96_nflows-8_ngroup-24-nearlyevery-99-nearlysize-99-nlayers-8_dilations-1-1-2-2-2-2-4-4_nchannels_96-kernelsize-5-lr-0.00100_seed-2019")
print(test_loss)
print(test_mse)

for i in range(5):
	context1s.append(model.generate(context1).cpu())
	context2s.append(model.generate(context2).cpu())
	context3s.append(model.generate(context3).cpu())
	context4s.append(model.generate(context4).cpu())
	context5s.append(model.generate(context5).cpu())
	context6s.append(model.generate(context6).cpu())
	context7s.append(model.generate(context7).cpu())
	context8s.append(model.generate(context8).cpu())
	context9s.append(model.generate(context9).cpu())



np.savetxt('./context1.csv', np.vstack([context1.cpu()]+[forecast1.cpu()]+context1s), delimiter=',')
np.savetxt('./context2.csv', np.vstack([context2.cpu()]+[forecast2.cpu()]+context2s), delimiter=',')
np.savetxt('./context3.csv', np.vstack([context3.cpu()]+[forecast3.cpu()]+context3s), delimiter=',')
np.savetxt('./context4.csv', np.vstack([context4.cpu()]+[forecast4.cpu()]+context4s), delimiter=',')
np.savetxt('./context5.csv', np.vstack([context5.cpu()]+[forecast5.cpu()]+context5s), delimiter=',')
np.savetxt('./context6.csv', np.vstack([context6.cpu()]+[forecast6.cpu()]+context6s), delimiter=',')
np.savetxt('./context7.csv', np.vstack([context7.cpu()]+[forecast7.cpu()]+context7s), delimiter=',')
np.savetxt('./context8.csv', np.vstack([context8.cpu()]+[forecast8.cpu()]+context8s), delimiter=',')
np.savetxt('./context9.csv', np.vstack([context9.cpu()]+[forecast9.cpu()]+context9s), delimiter=',')


