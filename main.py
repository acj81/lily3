import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms 
import matplotlib.pyplot as plt
# NN imports:
from torch import nn

# classes:

class basicNet(nn.Module):
	#
	def __init__(self):
		# parent init
		super().__init__()
		# transform we use - other ways of doing it but here bc readable:
		self.flatten = nn.Flatten()
		# define our layers and architecture - sequentially-layered in this case:
		self.layers = nn.Sequential(
			# no bias bc. input layer
			nn.Linear(28*28, 512, bias=False),
			# activation func w/ ReLU to compress inputs into usable range
			nn.ReLU(),
			# Linear activation layer in format (inp_size, out_size)
			nn.Linear(512, 512),
			nn.ReLU(),
			nn.Linear(512, 512),
			nn.ReLU(),
			# output layers - size 10 bc. 10 classes of images:
			nn.Linear(512, 10),
		)
		 
	# 
	def forward(self, x):
		''' given input x, propagate forward and get output - NOT MEANT TO CALL DIRECTLY'''
		# transform input by flattening:
		x = self.flatten(x)
		# propagate forward from input:
		out = self.layers(x)
		return out

	def train_on(self, dataloader, loss_fn, optimizer, epochs=1):
		''' given input dataset in dataloader, train on that dataset and update parameters, returning nothing'''
		# get size of dataset:
		size = len(dataloader)
		# set in training mode - good practice:
		self.train()
		# handle epochs if specified:
		for i in range(epochs):
			# enumerate for each example - technically don't need but cool:
			for batch_num, (inp, exp_out) in enumerate(dataloader):
				# propagate forward, then calculate loss:
				out = self(inp)
				example_loss = loss_fn(out, exp_out)

				# BACKPROPAGATION - actual training here:
				example_loss.backward() 
				optimizer.step()
				optimizer.zero_grad()
			
				# showing items:
				print(f"training on epoch / batch: {i + 1} / {batch_num}")

	def test_on(self, dataloader, loss_fn):
		''' given dataset, iterate through and test it: '''
	
		# prevent any gradients from being calculated - don't need to, just testing;
		with torch.no_grad():
			# need num. examples not batches:
			num_examples = len(dataloader.dataset)
			# store num. examples correctly classified, total loss:
			accuracy = 0
			avg_loss = 0
			# iterate through test examples:
			for i, (inp, exp_out) in enumerate(dataloader):
				# forward propagate, get loss for this example:
				out = self(inp)
				example_loss = loss_fn(out, exp_out).item()
				# get whether or not model was right (accuracy), how far off it was (loss):
				# whether model right for this example - convert to 1 or 0 with extra methods:
				accuracy += (out.argmax(1) == exp_out).type(torch.float).sum().item()
				# how far off it was;
				avg_loss += example_loss

				# print current batch num:
				print(f"testing batch: {i}")
				
			# get averages using size:
			accuracy /= num_examples
			avg_loss /= num_examples
			#
			return (accuracy, avg_loss)

# testing section:

if __name__ == "__main__":
	#
	print("loaded as main, testing...")
	#res = main()
	
	# create dataloader, then use to import dataset:
	training_data = datasets.FashionMNIST(
		root="data",
		train=True,
		download=True,
		transform=transforms.ToTensor()
	)
	
	# test dataset:
	testing_data = datasets.FashionMNIST(
		root="data",
		train=False,
		download=True,
		transform=transforms.ToTensor()
	)

	# loaders for datasets - processes data further for use:
	train_loader = DataLoader(training_data, batch_size=64, shuffle=True)
	test_loader = DataLoader(training_data, batch_size=64, shuffle=True)

	# test, show some actual data:
	head = training_data[0]
	tail = testing_data[1]
	print(head)
	print(tail)

	# we're using CPU:
	device = "cpu"

	# test instance of our NN:
	test_nn = basicNet()
	# out = test_nn.forward()
	#print(out)
	print(test_nn)

	# move it to our CPU, then pass some random input data:
	test_nn.to(device)
	inp = torch.rand(1, 28, 28, device=device)
	out = test_nn(inp)
	print(inp)
	print(out)


	# hyperparameters:
	epochs = 10
	batch_size=64
	learning_rate = 0.15
	# loss function:
	loss_fn = nn.CrossEntropyLoss()
	# optimizer - pass params of test nn:
	optim = torch.optim.SGD(test_nn.parameters(), lr=learning_rate)
	

	# test before training:
	starting_loss_test = test_nn.test_on(test_loader, loss_fn)
	starting_loss_train = test_nn.test_on(train_loader, loss_fn)
	
	# call training func:
	test_nn.train_on(train_loader, loss_fn, optim, epochs)

	# test after training:
	end_loss_test = test_nn.test_on(test_loader, loss_fn)
	end_loss_train = test_nn.test_on(train_loader, loss_fn)

	# print:
	print(f"before training: \ntest_data:{starting_loss_test} \ntrain_data:{starting_loss_train} \n ... after training: \ntest_data:{end_loss_test} \ntrain_data: {end_loss_train}")

	# save for later use to avoid having to re-train:
	torch.save(test_nn.state_dict(), 'model_weights.pth')
