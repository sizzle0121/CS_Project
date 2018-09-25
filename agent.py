import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from model import *
from utils import ReplayBuffer
import random
import numpy as np
import os

LongTensor = torch.LongTensor
FloatTensor = torch.FloatTensor


class DQN_agent:
	def __init__(self, N_ACT, history_m, lr, epsilon, epsilon_bound, gamma, replace_iter, batch_size, buffer_size):
		self.N_ACT = N_ACT
		self.history_m = history_m
		self.lr = lr
		self.epsilon = epsilon
		self.epsilon_bound = epsilon_bound
		self.gamma = gamma
		self.replace_iter = replace_iter
		self.iter = 0
		self.batch_size = batch_size
		self.begin_to_train = 0

		self.target_net = QNet(self.history_m, self.N_ACT).cuda()
		self.online_net = QNet(self.history_m, self.N_ACT).cuda()
		if os.path.isfile('target_net.pth'):
			self.target_net.load_state_dict(torch.load('target_net.pth'))
		if os.path.isfile('online_net.pth'):
			self.online_net.load_state_dict(torch.load('online_net.pth'))
		self.replay_buffer = ReplayBuffer(buffer_size)

		self.optimizer = optim.RMSprop(self.online_net.parameters(), lr = self.lr, alpha = 0.95, eps = 0.01, momentum = 0.0)
		self.loss_func = nn.MSELoss().cuda()


	def store_transition(self, transition):
		self.replay_buffer.store(transition)


	def select_action(self, state):
		if self.epsilon < random.uniform(0, 1):
			value, action = self.online_net(state).data.max(1)
			action = action.cpu().numpy()
			value = value.cpu().numpy()
		else:
			action = np.array([random.randrange(self.N_ACT)])
			value = [-12345]
		return value, action


	def train(self, mode):
		if self.begin_to_train < 50000:
			self.begin_to_train += 1
			return -12345, -12345

		if self.iter == self.replace_iter:
			self.target_net.load_state_dict(self.online_net.state_dict())
			self.iter = 0

		batch_transition = self.replay_buffer.sample(self.batch_size)
		state, action, reward, next_state, done = zip(*batch_transition)

		state = FloatTensor(np.array(state))
		action = LongTensor(np.array(action))
		reward = FloatTensor(np.array(reward))
		next_state = FloatTensor(np.array(next_state))


		state = Variable(state).cuda()
		action = Variable(action).cuda()
		reward = Variable(reward).cuda()
		next_state = Variable(next_state).cuda()
		

		Q = self.online_net(state).gather(1, action.view(-1, 1))
		if mode == 'DQN':
			Q_ = Variable(self.target_net(next_state).data.max(1)[0])
			y = reward + self.gamma * Q_
		elif mode == 'DDQN':
			online_act = Variable(self.online_net(next_state).data.max(1)[1]).cuda()
			Q_ = self.target_net(next_state).gather(1, online_act.view(-1, 1)).detach()
			y = reward.view(-1, 1) + self.gamma * Q_
		

		for i in range(len(done)):
			if done[i]:
				y[i] = reward[i]

		loss = self.loss_func(Q, y.view(-1, 1))
		self.optimizer.zero_grad()
		loss.backward()
		grad_norm = nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm = 10.0)
		self.optimizer.step()

		self.iter += 1
		self.epsilon = self.epsilon - 2*(1e-6) if self.epsilon > self.epsilon_bound else self.epsilon_bound
		return loss.item(), grad_norm.data.cpu().numpy()

	def save_param(self, episode):
		torch.save(self.target_net.state_dict(), 'target_net%d.pth'%(episode))
		torch.save(self.online_net.state_dict(), 'online_net%d.pth'%(episode))


