"""
	The file contains the PPO class to train with.
	NOTE: All "ALG STEP"s are following the numbers from the original PPO pseudocode.
			It can be found here: https://spinningup.openai.com/en/latest/_images/math/e62a8971472597f4b014c2da064f636ffe365ba3.svg
"""

import gym

import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam
#For continuous actions
from torch.distributions import MultivariateNormal
#For discrete action_space
from torch.distributions import Categorical
from network import FeedForwardActorNN, FeedForwardCriticNN
import sys
from cbf_clf_helper import clf_control, cbf_control

#Integrating tensorboard
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

class PPO:
	"""
		This is the PPO class we will use as our model in main.py
	"""
	def __init__(self, env, **hyperparameters):
		"""
			Initializes the PPO model, including hyperparameters.

			Parameters:
				policy_class - the policy class to use for our actor/critic networks.
				env - the environment to train on.
				hyperparameters - all extra arguments passed into PPO that should be hyperparameters.

			Returns:
				None
		"""
		# Make sure the environment is compatible with our code
		assert(type(env.observation_space) == gym.spaces.Box)
		# Makeassert(type(env.action_space) == gym.spaces.Box)

		# Initialize hyperparameters for training with PPO
		self._init_hyperparameters(hyperparameters)

		# Extract environment information
		self.env = env
		self.obs_dim = env.observation_space.shape[0]
		if self.discrete:
			self.act_dim = env.action_space.n
		else:
			self.act_dim = env.action_space.shape[0] #env.action_space.n #env.action_space.shape[0]

		 # Initialize actor and critic networks
		self.actor = FeedForwardActorNN(self.obs_dim, self.act_dim,self.discrete) 
		actor_model = 'ppo_actorKinematicBicycleGymLane.pth'
		policy = FeedForwardActorNN(5, 2,False)
		policy.load_state_dict(torch.load(actor_model))
		actor_model = policy
		#print(f'model =========== {self.actor}')                 	# ALG STEP 1
		self.critic = FeedForwardCriticNN(self.obs_dim, 1)
		#print(f'critic =========== {self.critic}') 

		# Initialize optimizers for actor and critic
		self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
		self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

		# Initialize the covariance matrix used to query the actor for actions
		self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.05)
		self.cov_mat = torch.diag(self.cov_var)
		self.obs_count = 0
		self.index_count = 0

		# This logger will help us with printing out summaries of each iteration
		self.logger = {
			't_so_far': 0,          # timesteps so far
			'i_so_far': 0,          # iterations so far
			'batch_lens': [],       # episodic lengths in batch
			'batch_rews': [],       # episodic returns in batch
			'batch_infractions': [],   # Episodic returns in a neural network
			'actor_losses': [],     # losses of actor network in current iteration
			'actor_network' : 0,	# Actor network
		}

	def learn(self, env_name,failure_observations,subpolicy):
		"""
			Train the actor and critic networks. Here is where the main PPO algorithm resides.

			Parameters:
				total_timesteps - the total number of timesteps to train for

			Return:
				None
		"""
		print(f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, ", end='')
		print(f"{self.timesteps_per_batch} timesteps per batch for a total of {self.training_step} iterations")
		t_so_far = 0 # Timesteps simulated so far
		i_so_far = 0 # Iterations ran so far
		while i_so_far < self.training_step:                                                                       # ALG STEP 2
			# Autobots, roll out (just kidding, we're collecting our batch simulations here)
			batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout(subpolicy,failure_observations)      # ALG STEP 3

			# Calculate how many timesteps we collected this batch
			t_so_far += np.sum(batch_lens)

			# Increment the number of iterations
			i_so_far += 1

			# Logging timesteps so far and iterations so far
			self.logger['t_so_far'] = t_so_far
			self.logger['i_so_far'] = i_so_far

			# Calculate advantage at k-th iteration
			V, _ = self.evaluate(batch_obs, batch_acts)
			A_k = batch_rtgs - V.detach()                                                                       # ALG STEP 5

			# One of the only tricks I use that isn't in the pseudocode. Normalizing advantages
			# isn't theoretically necessary, but in practice it decreases the variance of 
			# our advantages and makes convergence much more stable and faster. I added this because
			# solving some environments was too unstable without it.
			A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

			# This is the loop where we update our network for some n epochs
			for _ in range(self.n_updates_per_iteration):                                                       # ALG STEP 6 & 7
				# Calculate V_phi and pi_theta(a_t | s_t)
				V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

				# Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
				# NOTE: we just subtract the logs, which is the same as
				# dividing the values and then canceling the log with e^log.
				# For why we use log probabilities instead of actual probabilities,
				# here's a great explanation: 
				# https://cs.stackexchange.com/questions/70518/why-do-we-use-the-log-in-gradient-based-reinforcement-algorithms
				# TL;DR makes gradient ascent easier behind the scenes.
				ratios = torch.exp(curr_log_probs - batch_log_probs)

				# Calculate surrogate losses.
				#print(f'A_k======================={A_k}')
				surr1 = ratios * A_k
				#print(f'surr1======================={surr1}')
				surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k
				#print(f'surr2======================={surr2}')

				# Calculate actor and critic losses.
				# NOTE: we take the negative min of the surrogate losses because we're trying to maximize
				# the performance function, but Adam minimizes the loss. So minimizing the negative
				# performance function maximizes it.
				actor_loss = (-torch.min(surr1, surr2)).mean()
				#print(f'actor_loss======================={actor_loss}')
				critic_loss = nn.MSELoss()(V, batch_rtgs)

				# Calculate gradients and perform backward propagation for actor network
				self.actor_optim.zero_grad()
				actor_loss.backward(retain_graph=True)
				self.actor_optim.step()

				# Calculate gradients and perform backward propagation for critic network
				self.critic_optim.zero_grad()
				critic_loss.backward()
				self.critic_optim.step()

				# Log actor loss
				self.logger['actor_losses'].append(actor_loss.detach())
				self.logger['actor_network'] = self.actor

			# Print a summary of our training so far
			self._log_summary()

			# Save our model if it's time
			if i_so_far % self.save_freq == 0:
				if subpolicy:
					torch.save(self.actor.state_dict(), './ppo_actor_subpolicy'+env_name+'.pth')
					torch.save(self.critic.state_dict(), './ppo_critic_subpolicy'+env_name+'.pth')
				else:
					torch.save(self.actor.state_dict(), './ppo_actor'+env_name+'.pth')
					torch.save(self.critic.state_dict(), './ppo_critic'+env_name+'.pth')

	def rollout(self,subpolicy,failure_observations):
		"""
			This is where we collect the batch of data
			from simulation. Since this is an on-policy algorithm, we'll need to collect a fresh batch
			of data each time we iterate the actor/critic networks.

			Parameters:
				None

			Return:
				batch_obs - the observations collected this batch. Shape: (number of timesteps, dimension of observation)
				batch_acts - the actions collected this batch. Shape: (number of timesteps, dimension of action)
				batch_log_probs - the log probabilities of each action taken this batch. Shape: (number of timesteps)
				batch_rtgs - the Rewards-To-Go of each timestep in this batch. Shape: (number of timesteps)
				batch_lens - the lengths of each episode this batch. Shape: (number of episodes)
		"""
		# Batch data. For more details, check function header.
		batch_obs = []
		batch_acts = []
		batch_log_probs = []
		batch_rews = []
		batch_rtgs = []
		batch_lens = []
		batch_infractions = []

		# Episodic data. Keeps track of rewards per episode, will get cleared
		# upon each new episode
		ep_rews = []

		t = 0 # Keeps track of how many timesteps we've run so far this batch

		# Keep simulating until we've run more than or equal to specified timesteps per batch
		while t < self.timesteps_per_batch:
			act_list = []
			ep_rews = [] # rewards collected per episode
			# Reset the environment. sNote that obs is short for observation. 
			obs = self.env.reset()
			#print(f'obs reset ============= {obs}')
			done = False
			count_infractions = 0
			count_infractions_acc = 0
			count_infractions_steer = 0

			# Run an episode for a maximum of max_timesteps_per_episode timesteps
			for ep_t in range(self.max_timesteps_per_episode):
				a_predicted_clf = clf_control(self.env.v_ego)
				delta, target_id, crosstrack_error = self.env.car.tracker.stanley_control(self.env.x_ego, self.env.y_ego, self.env.yaw_ego, self.env.v_ego, self.env.delta_ego)
				# If render is specified, render the environment
				if self.render:
					self.env.render()

				t += 1 # Increment timesteps ran this batch so far

				# Track observations in this batch
				batch_obs.append(obs)

				# Calculate action and make a step in the env. 
				# Note that rew is short for reward.
				if self.discrete:
					action, log_prob = self.get_action_discrete(obs)
				else:
					action, log_prob = self.get_action(obs) #self.get_action_discrete(obs)
				#print(f'action chosen =============== {action}')
				if(abs(round(float(action[0]),1))<abs(round(float(a_predicted_clf),1))):
					count_infractions_acc = count_infractions_acc+1
				if(abs(round(float(action[1]),1)) < abs(round(float(delta),1))-0.2):
					#print(f'After rounding =============== {round(float(action_net[1]),1)} ====== {round(float(action[1]),1)}')
					count_infractions_steer = count_infractions_steer+1
				obs, rew, done, info = self.env.step(action)
				count_infractions = count_infractions_acc+count_infractions_steer


				# Track recent reward, action, and action log probability
				ep_rews.append(rew)
				batch_acts.append(action)
				batch_log_probs.append(log_prob)
				act_list.append(info)

				# If the environment tells us the episode is terminated, break
				if done:
					break

			# Track episodic lengths and rewards
			#self.env.render(act_list)
			batch_lens.append(ep_t + 1)
			batch_rews.append(ep_rews)
			batch_infractions.append(count_infractions)

		# Reshape data as tensors in the shape specified in function description, before returning
		batch_obs = torch.tensor(batch_obs, dtype=torch.float)
		#print(f'batch_acts =============== {batch_acts}')
		#For discrete state space
		if self.discrete:
			batch_acts = torch.tensor(batch_acts, dtype=torch.long).view(-1,)
		else:
			batch_acts = torch.tensor(batch_acts, dtype=torch.float) #torch.tensor(batch_acts, dtype=torch.long).view(-1,)
		#print(f'batch_acts =============== {batch_acts}')
		batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
		batch_rtgs = self.compute_rtgs(batch_rews)                                                              # ALG STEP 4

		# Log the episodic returns and episodic lengths in this batch.
		self.logger['batch_rews'] = batch_rews
		self.logger['batch_lens'] = batch_lens
		self.logger['batch_infractions'] = batch_infractions

		return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

	def compute_rtgs(self, batch_rews):
		"""
			Compute the Reward-To-Go of each timestep in a batch given the rewards.

			Parameters:
				batch_rews - the rewards in a batch, Shape: (number of episodes, number of timesteps per episode)

			Return:
				batch_rtgs - the rewards to go, Shape: (number of timesteps in batch)
		"""
		# The rewards-to-go (rtg) per episode per batch to return.
		# The shape will be (num timesteps per episode)
		batch_rtgs = []

		# Iterate through each episode
		for ep_rews in reversed(batch_rews):

			discounted_reward = 0 # The discounted reward so far

			# Iterate through all rewards in the episode. We go backwards for smoother calculation of each
			# discounted return (think about why it would be harder starting from the beginning)
			for rew in reversed(ep_rews):
				discounted_reward = rew + discounted_reward * self.gamma
				batch_rtgs.insert(0, discounted_reward)

		# Convert the rewards-to-go into a tensor
		batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

		return batch_rtgs


	# Probability sampling for discrete actions
	def get_action_discrete(self, obs):
		#print(f'obs ================== {obs}')
		mean = self.actor(obs)
		#print(f'mean ================== {mean}')

		dist = Categorical(mean)

		#print(f'dist ================== {dist}')

		action = dist.sample()

		log_prob = dist.log_prob(action)
		#print(f'action ====== {action} ========= {log_prob}')

		return action.detach().numpy().item(), log_prob.detach().item()


	def get_action(self, obs):
		"""
			Queries an action from the actor network, should be called from rollout.

			Parameters:
				obs - the observation at the current timestep

			Return:
				action - the action to take, as a numpy array
				log_prob - the log probability of the selected action in the distribution
		"""
		# Query the actor network for a mean action
		mean = self.actor(obs)

		# Create a distribution with the mean action and std from the covariance matrix above.
		# For more information on how this distribution works, check out Andrew Ng's lecture on it:
		# https://www.youtube.com/watch?v=JjB58InuTqM
		dist = MultivariateNormal(mean, self.cov_mat)

		# Sample an action from the distribution
		action = dist.sample()

		# Calculate the log probability for that action
		log_prob = dist.log_prob(action)

		# Return the sampled action and the log probability of that action in our distribution
		return action.detach().numpy(), log_prob.detach()

	def evaluate(self, batch_obs, batch_acts):
		"""
			Estimate the values of each observation, and the log probs of
			each action in the most recent batch with the most recent
			iteration of the actor network. Should be called from learn.

			Parameters:
				batch_obs - the observations from the most recently collected batch as a tensor.
							Shape: (number of timesteps in batch, dimension of observation)
				batch_acts - the actions from the most recently collected batch as a tensor.
							Shape: (number of timesteps in batch, dimension of action)

			Return:
				V - the predicted values of batch_obs
				log_probs - the log probabilities of the actions taken in batch_acts given batch_obs
		"""
		# Query critic network for a value V for each batch_obs. Shape of V should be same as batch_rtgs
		V = self.critic(batch_obs).squeeze()

		# Calculate the log probabilities of batch actions using most recent actor network.
		# This segment of code is similar to that in get_action()
		mean = self.actor(batch_obs)
		if self.discrete:
			dist = Categorical(mean)
		else:
			dist = MultivariateNormal(mean, self.cov_mat)
		#For discrete actions
		#dist = Categorical(mean)
		log_probs = dist.log_prob(batch_acts)

		# Return the value vector V of each observation in the batch
		# and log probabilities log_probs of each action in the batch
		return V, log_probs

	def _init_hyperparameters(self, hyperparameters):
		"""
			Initialize default and custom values for hyperparameters

			Parameters:
				hyperparameters - the extra arguments included when creating the PPO model, should only include
									hyperparameters defined below with custom values.

			Return:
				None
		"""
		# Initialize default values for hyperparameters
		# Algorithm hyperparameters
		self.timesteps_per_batch = 4800                 # Number of timesteps to run per batch
		self.max_timesteps_per_episode = 1600           # Max number of timesteps per episode
		self.n_updates_per_iteration = 5                # Number of times to update actor/critic per iteration
		self.lr = 0.005                                 # Learning rate of actor optimizer
		self.gamma = 0.95                               # Discount factor to be applied when calculating Rewards-To-Go
		self.clip = 0.2                                 # Recommended 0.2, helps define the threshold to clip the ratio during SGA

		# Miscellaneous parameters
		self.render = False                             # If we should render during rollout
		self.save_freq = 10                             # How often we save in number of iterations
		self.seed = None                                # Sets the seed of our program, used for reproducibility of results
		self.discrete = False							# Sets the type of environment to discrete or continuous
		self.training_step = 200						# Sets the number of trainig step

		# Change any default values to custom values for specified hyperparameters
		for param, val in hyperparameters.items():
			exec('self.' + param + ' = ' + str(val))

		# Sets the seed if specified
		if self.seed != None:
			# Check if our seed is valid first
			assert(type(self.seed) == int)

			# Set the seed 
			torch.manual_seed(self.seed)
			print(f"Successfully set seed to {self.seed}")

	def _log_summary(self):
		"""
			Print to stdout what we've logged so far in the most recent batch.

			Parameters:
				None

			Return:
				None
		"""
		# Calculate logging values. I use a few python shortcuts to calculate each value
		# without explaining since it's not too important to PPO; feel free to look it over,
		# and if you have any questions you can email me (look at bottom of README)
		t_so_far = self.logger['t_so_far']
		i_so_far = self.logger['i_so_far']
		avg_ep_lens = np.mean(self.logger['batch_lens'])
		avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])
		avg_actor_loss = np.mean([losses.float().mean() for losses in self.logger['actor_losses']])
		avg_ep_infractions = np.mean([np.sum(ep_inf) for ep_inf in self.logger['batch_infractions']])
		actor_model = self.logger['actor_network']

		# Round decimal places for more aesthetic logging messages
		avg_ep_lens = str(round(avg_ep_lens, 2))
		avg_ep_rews = str(round(avg_ep_rews, 2))
		avg_ep_infractions = str(round(avg_ep_infractions, 2))
		avg_actor_loss = str(round(avg_actor_loss, 5))

		writer.add_scalar("Average Episodic Return", int(float(avg_ep_rews)), t_so_far)
		writer.add_scalar("Average actor Loss", int(float(avg_actor_loss)), t_so_far)
		writer.add_scalar("Average Infractions", int(float(avg_ep_infractions)), t_so_far)
		# Tracking the weight of the network
		for name, param in actor_model.named_parameters():
			if 'weight' in name:
				writer.add_histogram(name, param.detach().numpy(), t_so_far)

		# Print logging statements
		print(flush=True)
		print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
		print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
		print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
		print(f"Average Episodic Infractions : {avg_ep_infractions}", flush=True)
		print(f"Average Loss: {avg_actor_loss}", flush=True)
		print(f"Timesteps So Far: {t_so_far}", flush=True)
		print(f"------------------------------------------------------", flush=True)
		print(flush=True)

		# Reset batch-specific logging data
		self.logger['batch_lens'] = []
		self.logger['batch_rews'] = []
		self.logger['actor_losses'] = []


def test(env, actor_model, is_discrete):
	"""
		Tests the model.
		Parameters:
			env - the environment to test the policy on
			actor_model - the actor model to load in
		Return:
			None
	"""
	print(f"Testing {actor_model}", flush=True)

	# If the actor model is not specified, then exit
	if actor_model == '':
		print(f"Didn't specify model file. Exiting.", flush=True)
		sys.exit(0)

	# Extract out dimensions of observation and action spaces
	obs_dim = env.observation_space.shape[0]
	if is_discrete:
		act_dim = env.action_space.n
	else:
		act_dim = env.action_space.shape[0] #env.action_space.n #env.action_space.shape[0]

	# Build our policy the same way we build our actor model in PPO
	policy = FeedForwardActorNN(obs_dim, act_dim,is_discrete)

	# Load in the actor model saved by the PPO algorithm
	policy.load_state_dict(torch.load(actor_model))
	

	# Evaluate our policy with a separate module, eval_policy, to demonstrate
	# that once we are done training the model/policy with ppo.py, we no longer need
	# ppo.py since it only contains the training algorithm. The model/policy itself exists
	# independently as a binary file that can be loaded in with torch.
	eval_policy(policy=policy, env=env, render=True, is_discrete=is_discrete)
	


