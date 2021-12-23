import numpy as np
import pickle
#Import the environment file
from env_turn import KinematicBicycleGymTurn
from env_lane import KinematicBicycleGymLane
from env_cut_in import KinematicBicycleGymCutIn
from env_acc import KinematicBicycleGymACC
import torch
import random
from cbf_clf_helper import clf_control, cbf_control
#Integrating tensorboard
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
from ppoPolicyTraining import PPO
from ppoPolicyTrainingGuided import PPOGuided
from network import FeedForwardActorNN, FeedForwardCriticNN
import argparse
import yaml



#==================================================Code to plot Tensorboard Graphs =====================================#
	

def _log_execution(count,speed,absolute_error,distance_fc,distance_sc):
	"""
		Print to stdout what we've logged so far in the most recent batch.

		Parameters:
			count,speed,absolute_error,distance_fc,distance_sc

		Return:
			None
	"""
	writer.add_scalar("Speed", int(float(speed)), count)
	writer.add_scalar("Absolute Error", int(float(absolute_error)), count)
	writer.add_scalar("Distance_fc", int(float(distance_fc)), count)
	writer.add_scalar("Distance_sc", int(float(distance_sc)), count)

#==================================================Code to Test the Policies =====================================#


def test_ppo(env,actor_model):
	done = False
	#actor_model = 'ppo_actorKinematicBicycleGymCutIn.pth'
	#actor_model = 'OriginalPPOCutIn/ppo_actorKinematicBicycleGymCutIn.pth'
	#actor_model = 'GuidedAtEveryStepCutIN/ppo_actorKinematicBicycleGymCutIn.pth'
	#actor_model = 'BestCutInGuidedPolicyOur/ppo_actorKinematicBicycleGymCutIn.pth'
	policy = FeedForwardActorNN(5, 2,False)
	policy.load_state_dict(torch.load(actor_model))
	observation = env.reset()
	action_list = []
	itercount = 0
	score = 0
	count_inf = 0
	count_inf_steer = 0
	while not done:
		a_predicted_clf = clf_control(env.v_ego)
		delta, target_id, crosstrack_error = env.car.tracker.stanley_control(env.x_ego, env.y_ego, env.yaw_ego, env.v_ego, env.delta_ego)
		state = torch.tensor(observation, dtype=torch.float)
		target_index, dx, dy, absolute_error = env.car.tracker.find_target_path_id(env.x_ego, env.y_ego, env.yaw_ego)
		yaw_error = env.car.tracker.calculate_yaw_term(target_index, env.yaw_ego)
		print(f'absolute_error ============= {absolute_error}===yaw_error=={yaw_error}========speed ===== {env.v_ego} ====== ')
		_log_execution(itercount,env.v_ego,absolute_error,env.distance_fc,env.distance_sc)
		a_net = policy(state).detach().numpy() #model.predict(state) #agent.choose_action_test(state) model.predict(state)
		throttle = a_net[0]
		delta = a_net[1]
		if(abs(round(float(a_net[0]),1)))<abs(round(float(a_predicted_clf),1)):
			count_inf=count_inf+1
		if(abs(round(float(a_net[1]),1))<abs(round(float(delta),1))):
			count_inf_steer = count_inf_steer+1
		#action = [a_net[0][0],0.0]
		action = [throttle,delta]
		#print(f'state ============= {state}')
		observation_, reward_agent, done, info = env.step(action)
		print(f'action ============= {info}====={count_inf+count_inf_steer}')
		action = [info[0],info[1]]
		action_list.append(info)
		score += reward_agent
		observation = observation_
		itercount = itercount+1
		#env.render()

	print(f'score =========={score}')
	env.render(action_list)





if __name__ == "__main__":

	#=============================== Environment and Hyperparameter Configuration Start ================================#
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('--env', dest='env', action='store_true', help='environment_name')
	parser.add_argument('--traintype', dest='traintype', action='store_true', help='Type of training 1. Vanilla 2. Guided 3.S2RL')
	parser.add_argument('--train', dest='train', action='store_true', help='train model')
	parser.add_argument('--test', dest='test', action='store_true', help='test model')
	parser.add_argument('--actor', dest='actor', action='store_true', help='Actor Model')
	parser.add_argument('--critic', dest='critic', action='store_true', help='Critic Model')
	args = parser.parse_args()
	# Default Configurations
	actor_model = None
	critic_model = None
	if args.env:
		env_name = args.env
	else:
		env_name = 'KinematicBicycleGymACC'
	if args.traintype:
		traintype = args.traintype
	else:
		traintype = 'S2RL'
	if args.actor:
		actor_model = args.actor
	else:
		actor_model = 'ACCS2RLPPO/ppo_actorKinematicBicycleGymACC.pth'
	if args.critic:
		critic_model = args.critic
	else:
		critic_model = 'ACCS2RLPPO/ppo_criticKinematicBicycleGymACC.pth'

	#Load the hyperparameters

	#Create custom environment
	if(env_name=='KinematicBicycleGymACC'):
		env = KinematicBicycleGymACC()
	elif(env_name=='KinematicBicycleGymCutIn'):
		env = KinematicBicycleGymCutIn()
	elif(env_name=='KinematicBicycleGymLane'):
		env = KinematicBicycleGymLane()
	else:
		env = KinematicBicycleGymTurn()
	

	with open('hyperparameters.yml') as file:
		paramdoc = yaml.full_load(file)
	#=============================== Environment and Hyperparameter Configuration End ================================#
	#=============================== Original Policy Training Code Start ================================#
	if args.train:
		config_name = env_name
		if(traintype == 'PPO'):
			config_name = env_name + '-PPO'
		for item, param in paramdoc.items():
			if(str(item)==config_name):
				hyperparameters = param
				print(param) 
		if(traintype == 'PPO'):
			model = PPO(env=env, **hyperparameters)
		else:
			model = PPOGuided(env=env, **hyperparameters)
		model.learn(env_name, [], False)
	#=============================== Original Policy Training Code End ================================#
	#=============================== Policy Testing Code Start ==========================#
	if args.test:
		test_ppo(env,actor_model)
	#=============================== Policy Testing Code End ============================#
	
	#===============Training using Baselines=========================================================#
	'''n_actions = env.action_space.shape[-1]
	action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
	model = TD3("MlpPolicy", env,  tensorboard_log='TD3', verbose=1)
	model.learn(total_timesteps=10000, log_interval=1)
	model.save("td3_baseline")'''
	#===============Testing Baselines=========================================================#
	#test_gym(env)

	'''env_name = 'KinematicBicycleGymTurn'
	hyperparameters = {'timesteps_per_batch': 2048,
    'max_timesteps_per_episode': 500,
    'gamma': 0.99,
    'n_updates_per_iteration': 10,
    'lr': 2.5e-3,
    'clip': 0.2,
    'seed': 741,
    'training_step' : 500}
	model = PPO(env=env, **hyperparameters)
	model.learn(env_name, [], False)'''

	'''env_name = 'KinematicBicycleGymTurn'
	hyperparameters = {'timesteps_per_batch': 2048,
    'max_timesteps_per_episode': 500,
    'gamma': 0.99,
    'n_updates_per_iteration': 20,
    'lr': 2.5e-3,
    'clip': 0.2,
    'seed': 741,
    'training_step' : 250}
	model = PPOGuided(env=env, **hyperparameters)
	model.learn(env_name, [], False)'''

	'''env_name = 'KinematicBicycleGymLane'
	hyperparameters = {'timesteps_per_batch': 2048,
    'max_timesteps_per_episode': 500,
    'gamma': 0.99,
    'n_updates_per_iteration': 20,
    'lr': 2.5e-3,
    'clip': 0.2,
    'seed': 741,
    'training_step' : 350}
	model = PPOGuided(env=env, **hyperparameters)
	model.learn(env_name, [], False)'''

	'''env_name = 'KinematicBicycleGymLane'
	hyperparameters = {'timesteps_per_batch': 2048,
    'max_timesteps_per_episode': 500,
    'gamma': 0.99,
    'n_updates_per_iteration': 10,
    'lr': 2.5e-4,
    'clip': 0.2,
    'seed': 741,
    'training_step' : 2000}
	model = PPO(env=env, **hyperparameters)
	model.learn(env_name, [], False)'''

	'''env_name = 'KinematicBicycleGymCutIn'
	hyperparameters = {'timesteps_per_batch': 2048,
    'max_timesteps_per_episode': 500,
    'gamma': 0.99,
    'n_updates_per_iteration': 10,
    'lr': 2.5e-4,
    'clip': 0.2,
    'seed': 741,
    'training_step' : 2000}
	model = PPO(env=env, **hyperparameters)
	model.learn(env_name, [], False)'''

	'''env_name = 'KinematicBicycleGymCutIn'
	hyperparameters = {'timesteps_per_batch': 2048,
    'max_timesteps_per_episode': 550,
    'gamma': 0.99,
    'n_updates_per_iteration': 10,
    'lr': 2.5e-4,
    'clip': 0.1,
    'seed': 741,
    'training_step' : 600}
	model = PPOGuided(env=env, **hyperparameters)
	model.learn(env_name, [], False)'''

	'''env_name = 'KinematicBicycleGymACC'
	hyperparameters = {'timesteps_per_batch': 2048,
    'max_timesteps_per_episode': 500,
    'gamma': 0.99,
    'n_updates_per_iteration': 10,
    'lr': 2.5e-4,
    'clip': 0.2,
    'seed': 741,
    'training_step' : 1000}
	model = PPO(env=env, **hyperparameters)
	model.learn(env_name, [], False)'''

	'''env_name = 'KinematicBicycleGymACC'
	hyperparameters = {'timesteps_per_batch': 2048,
    'max_timesteps_per_episode': 500,
    'gamma': 0.99,
    'n_updates_per_iteration': 20,
    'lr': 2.5e-3,
    'clip': 0.2,
    'seed': 741,
    'training_step' : 350}
	model = PPOGuided(env=env, **hyperparameters)
	model.learn(env_name, [], False)'''





