import gym
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torch.backends.cudnn as cudnn

from utils import *
from agent import *

import os
import numpy as np
import matplotlib.pyplot as plt
import csv

LongTensor = torch.LongTensor
FloatTensor = torch.FloatTensor

env = gym.make('Tennis-v0')
#env = NoopResetEnv(env)
#env = MaxAndSkipEnv(env)
#print(env.action_space.n)
#print(env.observation_space.shape[0])

N_ACT = env.action_space.n
#N_OBS = env.observation_space.shape[0]
history_m = 3
lr = 0.00025
epsilon = 1.0
epsilon_bound = 0.01
gamma = 0.99
replace_iter = 10000
batch_size = 32
buffer_size = 85000
EPISODE = 20000
TEST = False
skip_num = 3

agent = DQN_agent(N_ACT, history_m, lr, epsilon, epsilon_bound, gamma, replace_iter, batch_size, buffer_size)


def play_game(EPISODE):
	return_writer = csv.writer(open("./Return.csv", "w"))
	summary_writer = csv.writer(open("./training_log.csv", "w"))
	for episode in range(EPISODE):
		done = False
		R = 0
		num_step = 0
		pass_time = 0

		if TEST:
			obs = env.reset()
		else:
			obs = preprocess(np_to_pil(env.reset()))	
			obs = np.stack([obs]*3, axis = 0)
			
			AGENT_SCORE = 0
			OPPO_SCORE = 0
			AGENT_SCORE_per = 0
			OPPO_SCORE_per = 0
			SWITCH_SIDE = 1

		while not done:
			#env.render()
			if TEST:
				action = agent.select_action(Variable(torch.Tensor([obs])).cuda())
			else:
				with torch.no_grad():
					S = torch.Tensor(obs)
					MaxQ, action = agent.select_action(Variable(S.view(1, 3, 84, 84)).cuda())
			
			obs_buffer = []
			reward_buffer = 0.0
			org_reward_buffer = 0.0

			for frame_num in range(skip_num):
				obs_, reward, done, _ = env.step(action[0])
				obs_ = preprocess(np_to_pil(obs_))
				obs_buffer.append(obs_)
				if reward == -1:
					OPPO_SCORE += 1
					OPPO_SCORE_per += 1
				elif reward == 1:
					AGENT_SCORE += 1
					AGENT_SCORE_per += 1
				reward_buffer += reward
				if done:
					for i in range(skip_num - frame_num - 1):
						obs_buffer.append(obs_)
					break


			case = check_serve(action[0], pass_time)

			serve_reward = 0
			if case == -1:
				pass_time = -1
			elif case == 1:
				serve_reward = 1
				pass_time = -1
			elif case == 2:
				pass_time += 1
			elif case == 3:
				serve_reward = -1
				pass_time = -1

			if reward_buffer != 0:
				pass_time = 0

			if check_end(AGENT_SCORE_per, OPPO_SCORE_per):
				SWITCH_SIDE += 1
				AGENT_SCORE_per = 0
				OPPO_SCORE_per = 0

			org_reward_buffer += reward_buffer
			reward_buffer += 0.5 * serve_reward * (0 if SWITCH_SIDE%2 == 0 else 1)
			obs_ = np.stack(obs_buffer, axis = 0)

			if TEST:
				transition = [
					FloatTensor([obs]),
					LongTensor(action),
					FloatTensor([reward]),
					FloatTensor([obs_]),
					done
				]
			else:
				
				transition = [
					obs.reshape(3, 84, 84),#obs.view(1, 4, 84, 84),#np_obs,
					action,#LongTensor(action),
					reward_buffer,#FloatTensor([reward_buffer]),#np.array([reward]),
					obs_.reshape(3, 84, 84),#obs_.view(1, 4, 84, 84),#np_obs_,
					done
				]

			agent.store_transition(transition)
			loss, grad_norm = agent.train('DDQN')
			obs = obs_
			R += org_reward_buffer
			num_step += 1
			summary_writer.writerow([MaxQ[0], action[0], loss, grad_norm])
		print('Episode: %3d,\tReturn: %f,\tStep: %f' %(episode, R, num_step))
		print('OPPONENT: %f' %(OPPO_SCORE))
		print('AGENT   : %f' %(AGENT_SCORE))
		if episode % 200 == 0:
			agent.save_param(episode)
		return_writer.writerow([R, num_step])


if __name__ == "__main__":
	play_game(EPISODE)


