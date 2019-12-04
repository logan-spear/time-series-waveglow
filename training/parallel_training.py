import multiprocessing, os
from random import choice
import numpy as np
from waveglow_train import run_training
import threading
# from multiprocessing import Process
import pickle

def check_threads(threads):
	count = 0
	for i in reversed(range(len(threads))):
		if not threads[i].isAlive():
			del threads[i]
			count += 1
	
	return count

# def run_new(args):
	# os.system('python3 train.py %s' % args)
with open("./list_of_232_configs", "rb") as fp:
	test_list = pickle.load(fp)

threads = []
max_threads = 2

print("Starting %d threads" % max_threads)


epochs=250
batch_size=120
seed=2019
generate_per_epoch=False
generate_final_plots=True
checkpointing=True
use_gpu=False
n_channels=96
n_context_channels=96
rolling=True
datafile="./wind_power_data/wind_power_train.pickle"
valid_split=.2
small_subset=False
validation_patience=10


for t in range(max_threads):
	config = test_list.pop()
	threads.append(threading.Thread(target=run_training, args=(config, epochs, batch_size, seed, generate_per_epoch, generate_final_plots, checkpointing, use_gpu, n_channels, n_context_channels, rolling, datafile, valid_split, small_subset, validation_patience,)))

	threads[-1].start()

print("Finished starting %d threads" % max_threads)

while(len(test_list) > 0):
	count = check_threads(threads)
	if count > 0:
		for c in range(count):
			if len(test_list) == 0: break
		
			# threads.append(Thread(target=run_new, args=(new_args,)))
			# print("Starting thread for node %d" % next_node)
			# threads[-1].start()
			# num_tests-=1
			config = test_list.pop()
			threads.append(threading.Thread(target=run_training, args=(config, epochs, batch_size, seed, generate_per_epoch, generate_final_plots, checkpointing, use_gpu, n_channels, n_context_channels, rolling, datafile, valid_split, small_subset, validation_patience,)))

			threads[-1].start()
	else:
		continue
		
for t in threads:
	t.join()




# import multiprocessing, os
# from random import choice
# import numpy as np
# from waveglow_train import run_training
# from multiprocessing import Process
# import pickle

# def check_threads(threads):
#     count = 0
#     for i in reversed(range(len(threads))):
#         if not threads[i].isAlive():
#             del threads[i]
#             count += 1
	
#     return count

# # def run_new(args):
#   # os.system('python3 train.py %s' % args)
# with open("./list_of_232_configs", "rb") as fp:
#     test_list = pickle.load(fp)

# processes = []
# max_processes = 10

# print("Starting %d threads" % max_processes)


# epochs=250
# batch_size=120
# seed=2019
# generate_per_epoch=False
# generate_final_plots=True
# checkpointing=True
# use_gpu=False
# n_channels=96
# n_context_channels=96
# rolling=True
# datafile="./wind_power_data/wind_power_train.pickle"
# valid_split=.2
# small_subset=False
# validation_patience=10


# for p in range(max_processes):
#     config = test_list.pop()
#     processes.append(Process(target=run_training, args=(config, epochs, batch_size, seed, generate_per_epoch, generate_final_plots, checkpointing, use_gpu, n_channels, n_context_channels, rolling, datafile, valid_split, small_subset, validation_patience,)))

#     processes[-1].start()

# print("Finished starting %d processes" % max_processes)

# while(len(test_list) > 0):
#     count = check_processes(processes)
#     if count > 0:
#         for c in range(count):
#             if len(test_list) == 0: break
		
#             # threads.append(Thread(target=run_new, args=(new_args,)))
#             # print("Starting thread for node %d" % next_node)
#             # threads[-1].start()
#             # num_tests-=1
#             config = test_list.pop()
#             processes.append(Process(target=run_training, args=(config, epochs, batch_size, seed, generate_per_epoch, generate_final_plots, checkpointing, use_gpu, n_channels, n_context_channels, rolling, datafile, valid_split, small_subset, validation_patience,)))

#             processes[-1].start()
#     else:
#         continue
		
# for p in processes:
#     p.join()


