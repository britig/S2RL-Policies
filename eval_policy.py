"""
	This file is used only to evaluate our trained policy/actor after 
	training in main.py with ppo.py.
"""

import numpy as np
import torch

def _log_summary(ep_len, ep_ret, ep_num):
		"""
			Print to stdout what we've logged so far in the most recent episode.

			Parameters:
				None

			Return:
				None
		"""
		# Round decimal places for more aesthetic logging messages
		ep_len = str(round(ep_len, 2))
		ep_ret = str(round(ep_ret, 2))

		# Print logging statements
		print(flush=True)
		print(f"-------------------- Episode #{ep_num} --------------------", flush=True)
		print(f"Episodic Length: {ep_len}", flush=True)
		print(f"Episodic Return: {ep_ret}", flush=True)
		print(f"------------------------------------------------------", flush=True)
		print(flush=True)

'''
	Method for testing the learnt policy
	Needs to be terminated manually
'''
def rollout(policy, env, render, is_discrete):
	"""
		Returns a generator to roll out each episode given a trained policy and
		environment to test on. 

		Parameters:
			policy - The trained policy to test
			env - The environment to evaluate the policy on
			render - Specifies whether to render or not
		
		Return:
			A generator object rollout, or iterable, which will return the latest
			episodic length and return on each iteration of the generator.

		Note:
			If you're unfamiliar with Python generators, check this out:
				https://wiki.python.org/moin/Generators
			If you're unfamiliar with Python "yield", check this out:
				https://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do
	"""
	# Rollout until user kills process
	while True:
		obs = env.reset()
		done = False

		# number of timesteps so far
		t = 0

		# Logging data
		ep_len = 0            # episodic length
		ep_ret = 0            # episodic return
		traj = [obs]
		actions = []
		while not done:
			t += 1

			# Render environment if specified, off by default
			if render:
				env.render()

			# Query deterministic action from policy and run it
			if is_discrete:
				action = choose_best_action(obs,policy)
			else:
				action = policy(obs).detach().numpy()  #policy(obs).detach().numpy()
			obs, rew, done, _ = env.step(action)
			actions.append(action)
			traj.append(obs)
			# Sum all episodic rewards as we go along
			ep_ret += rew
			
		# Track episodic length
		ep_len = t
		'''if ep_ret < 200:
			print(f'Reward ========= {ep_ret}')
			print(f'traj start ========= {traj[0]}')
			print(f'Action ========= {actions}')'''

		# returns episodic length and return in this iteration
		yield ep_len, ep_ret


#For discrete actions [cartpole environment]
def choose_best_action(obs,policy):
	#print(f'obs ============ {obs}')
	y = policy(obs).squeeze()
	#print(f'y ============ {y}')
	action = torch.argmax(y)
	#print(f'action ============ {action}==== {action.item()}')

	return action.item()




#Function to display a single trajectory
def display(observation,policy,env,is_discrete):
	name = env.unwrapped.spec.id
	ep_ret = 0
	#this is specific to inverted pendulum environment
	if name == 'Pendulum-v0':
		env.env.state = observation[0:2]
		obs = observation
	else:
		env.env.state = observation
		obs = np.array(env.env.state)
	#obs = observation
	#print(f'obs ============= {obs}')
	traj = [obs]
	done = False
	iter = 0
	while not done:
		#env.render()
		if is_discrete:
			action = choose_best_action(obs,policy)
		else:
			action = policy(obs).detach().numpy()
		#print(f'action ============= {action}')
		obs, rew, done, _ = env.step(action)
		#print(f'obs x ============= {obs[0]}')
		'''if(env.env.hull.angle > 2 or env.env.hull.angle < -0.8):
			print(f'hull angle ============= {env.env.hull.angle}')'''
		traj.append(obs)
		#print(f'obs, action, rew, done ===={action} ===== {rew}==== {done}')

		# Sum all episodic rewards as we go along
		ep_ret += rew
		iter = iter+1
		
	print(f'Reward ============= {ep_ret}')
	return ep_ret, traj, iter


def eval_policy(policy, env, render=False, is_discrete=False):
	"""
		The main function to evaluate our policy with. It will iterate a generator object
		"rollout", which will simulate each episode and return the most recent episode's
		length and return. We can then log it right after. And yes, eval_policy will run
		forever until you kill the process. 

		Parameters:
			policy - The trained policy to test, basically another name for our actor model
			env - The environment to test the policy on
			render - Whether we should render our episodes. False by default.

		Return:
			None

		NOTE: To learn more about generators, look at rollout's function description
	"""
	# Rollout with the policy and environment, and log each episode's data
	for ep_num, (ep_len, ep_ret) in enumerate(rollout(policy, env, render, is_discrete)):
		_log_summary(ep_len=ep_len, ep_ret=ep_ret, ep_num=ep_num)

def collect_batches(policy, critic, env, render=False):
	'''for ep_num, (ep_len, ep_ret) in enumerate(collect_failure_traces(policy, critic, env, render)):
		_log_summary(ep_len=ep_len, ep_ret=ep_ret, ep_num=ep_num)
		if(ep_ret<0):
			break'''
	display(policy,critic,env,False)
