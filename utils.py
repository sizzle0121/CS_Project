import gym
import torch
import torchvision.transforms as transforms
import random
from PIL import Image
import numpy as np
from collections import deque


def np_to_pil(img):
	arr = img.astype(np.uint8)
	arr = arr[55:215,...]
	new_image = Image.fromarray(arr)
	return new_image

#(210, 160, 3)
#(168, 160) to crop off the plate
preprocess = transforms.Compose([
		transforms.Grayscale(num_output_channels = 1),
		transforms.Resize(size = (84, 84), interpolation = 1),
	])

GrayScale = transforms.Grayscale(num_output_channels = 1)


class ReplayBuffer:
	def __init__(self, CAPACITY):
		self.capacity = CAPACITY
		self.memory = []

	def store(self, transition):
		self.memory.append(transition)
		if len(self.memory) > self.capacity:
			del self.memory[0]

	def sample(self, batch_size):
		return random.sample(self.memory, batch_size)

	def __len__(self):
		return len(self.memory)



def check_serve(action, pass_time):
	FIRE = [1, 10, 11, 12, 13, 14, 15, 16, 17]
	if pass_time < 0:
		return -1
	elif action in FIRE:
		return 1
	else:
		return 2 if pass_time < 10 else 3

def check_end(AGENT, OPPO):
	if AGENT >= 4 and AGENT - OPPO >= 2:
		return True
	elif OPPO >= 4 and OPPO - AGENT >= 2:
		return True
	else:
		return False

