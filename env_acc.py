'''
Project : S2RL
Author : Briti Gangopadhyay
This file contains the environment for simulating a Kinematic Bicycle Model
Class Names
Simulation
Path
Car
KinematicBicycleGym : Gym wrapper for the Kinematic Bicycle Model

'''


import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random as rand

from kinematic_model import KinematicBicycleModel
from matplotlib.animation import FuncAnimation
from libs.stanley_controller import StanleyController
from libs.car_description import Description
from libs.cubic_spline_interpolator import generate_cubic_spline
from cbf_clf_helper import clf_control,cbf_control
import math

from math import copysign, sin
import gym
from gym import spaces
from libs.normalise_angle import normalise_angle

global myvar
global action_list

class Simulation:

	def __init__(self):

		fps = 50.0

		self.dt = 1/fps
		self.map_size = 40
		self.frames = 2500
		self.loop = False

class Path:

	def __init__(self):

		# Get the path information from waypoints.csv
		dir_path = 'data/waypointsacc1.csv'
		df = pd.read_csv(dir_path)

		self.x_path = df['X-axis'].values
		self.y_path = df['Y-axis'].values

		#Path for fc vehicle
		dir_path_fc = 'data/waypointsfc.csv'
		df_fc = pd.read_csv(dir_path_fc)
		self.x_path_fc = df_fc['X-axis'].values
		self.y_path_fc = df_fc['Y-axis'].values

		#Path for sc vehicle
		dir_path_sc = 'data/waypointssc.csv'
		df_sc = pd.read_csv(dir_path_sc)
		self.x_path_sc = df_sc['X-axis'].values
		self.y_path_sc = df_sc['Y-axis'].values

		ds = 0.05

		self.px, self.py, self.pyaw, _ = generate_cubic_spline(self.x_path, self.y_path, ds)
		self.px_fc, self.py_fc, self.pyaw_fc, _ = generate_cubic_spline(self.x_path_fc, self.y_path_fc, ds)
		self.px_sc, self.py_sc, self.pyaw_sc, _ = generate_cubic_spline(self.x_path_sc, self.y_path_sc, ds)

class Car:

	def __init__(self, init_x, init_y, init_yaw, px, py, pyaw, dt):

		# Model parameters
		self.x = init_x
		self.y = init_y
		#print(f'self.x======={self.x}=====self.y==={self.y}')
		self.yaw = init_yaw
		self.v = 0.0
		self.delta = 0.0
		self.omega = 0.0
		self.L = 2.5
		self.max_steer = np.deg2rad(33)
		self.dt = dt
		self.c_r = 0.01
		self.c_a = 2.0

		# Tracker parameters
		self.px = px
		self.py = py
		self.pyaw = pyaw
		self.k = 8.0
		self.ksoft = 1.0
		self.kyaw = 0.01
		self.ksteer = 0.0
		self.crosstrack_error = None
		self.target_id = None

		# Description parameters
		self.length = 4.5
		self.width = 2.0
		self.rear2wheel = 1.0
		self.wheel_dia = 0.15 * 2
		self.wheel_width = 0.2
		self.tread = 0.7
		self.colour = 'black'

		self.tracker = StanleyController(self.k, self.ksoft, self.kyaw, self.ksteer, self.max_steer, self.L, self.px, self.py, self.pyaw)
		self.kbm = KinematicBicycleModel(self.L, self.max_steer, self.dt, self.c_r, self.c_a)

	def drive(self,throttle,delta):
		
		x_e, y_e, yaw_e, v_e, _, _ = self.kbm.kinematic_model(self.x, self.y, self.yaw, self.v, throttle, delta)
		self.x = x_e
		self.y = y_e
		self.yaw = yaw_e
		self.v = v_e

		#os.system('cls' if os.name=='nt' else 'clear')
		#print(f"Cross-track term: {self.crosstrack_error}")
		return x_e, y_e, yaw_e, v_e

	#For calculation of a lookahead step
	def drive_dummy(self,throttle,delta):
		
		x_e, y_e, yaw_e, v_e, _, _ = self.kbm.kinematic_model(self.x, self.y, self.yaw, self.v, throttle, delta)

		#os.system('cls' if os.name=='nt' else 'clear')
		#print(f"Cross-track term: {self.crosstrack_error}")
		return x_e, y_e, yaw_e, v_e


class KinematicBicycleGymACC(gym.Env):

	def __init__(self):
		super(KinematicBicycleGymACC, self).__init__()
		# Two actions one for steering another for acceleration
		# Observation space position x, position y, yaw, velocity
		self.observation_space = spaces.Box(-np.inf, np.inf, shape=(5,), dtype=np.float32)
		self.action_space = spaces.Box(-3.0, 3.0, (2,), dtype=np.float32)
		#self.action_space = spaces.Box(-1, +1, (2,), dtype=np.float32)
		#self.action_space = spaces.Box(low=np.array([-1.0,-0.6]), high=np.array([1.4,0.6]),shape=(2,), dtype=np.float32)


	def reset(self):
		#set render flag
		render = True
		self.sim = Simulation()

		#get the staring point of the path

		# =============== this section is for the ego vehicle ============================ #
		self.path = Path()
		self.x_ego = self.path.x_path[0] #Ego vehicle's position
		self.y_ego = self.path.y_path[0]#+rand.randint(-5,5) #Ego vehicle's position
		
		self.car = Car(self.x_ego , self.y_ego , self.path.pyaw[0], self.path.px, self.path.py, self.path.pyaw, self.sim.dt)
		self.desc = Description(self.car.length, self.car.width, self.car.rear2wheel, self.car.wheel_dia, self.car.wheel_width, self.car.tread, self.car.L)

		# =============== This section is for the front vehicles ============================ #

		self.x_fc = self.path.x_path_fc[0]
		self.y_fc = self.path.y_path_fc[0]

		self.car_fc = Car(self.x_fc , self.y_fc , self.path.pyaw_fc[0], self.path.px_fc, self.path.py_fc, self.path.pyaw_fc, self.sim.dt)
		self.desc_fc = Description(self.car_fc.length, self.car_fc.width, self.car_fc.rear2wheel, self.car_fc.wheel_dia, self.car_fc.wheel_width, self.car_fc.tread, self.car_fc.L)

		# ================= This section is for the side vehicles ========================= #

		self.x_sc = self.path.x_path_sc[0]
		self.y_sc = self.path.y_path_sc[0]

		self.car_sc = Car(self.x_sc , self.y_sc , self.path.pyaw_sc[0], self.path.px_sc, self.path.py_sc, self.path.pyaw_sc, self.sim.dt)
		self.desc_sc = Description(self.car_sc.length, self.car_sc.width, self.car_sc.rear2wheel, self.car_sc.wheel_dia, self.car_sc.wheel_width, self.car_sc.tread, self.car_sc.L)



		self.interval = self.sim.dt * 10**3


		self.state_max = np.hstack(
			(30,
			 50,
			 30,
			 30,
			 100))
		self.state_min = np.hstack(
			(0,
			 0,
			 0,
			 0,
			 0))
			
		#Environment variables
		self.x_target = self.path.x_path[-1]
		self.y_target = self.path.y_path[-1]
		#self.check_x_target = 30.0
		#self.check_y_target = 47.0
		self.v_ego = 0.0 #Ego vehicle's velocity

		self.delta_ego = 0.0
		self.v_desired = 20 #Desired Velocity
		self.v_lead = 0.0
		self.car_fc.v = self.v_lead
		self.v_sc = 0.0
		self.car_fc.v = self.v_sc 

		#Distance
		self.distance_fc = 10
		self.distance_sc = 10

		self.yaw_ego = self.path.pyaw[0]
		self.safe_distance = 100
		self.counter = 0
		self.delta = 0.0
		self.diff_target = np.sqrt((self.x_ego-self.x_target)**2 + (self.y_ego-self.y_target)**2)
		obs = self.feature_scaling(np.hstack((self.x_ego,self.y_ego,self.v_ego, (self.v_desired-self.v_ego),self.diff_target)))
		return np.array(obs, dtype=np.float32)


	'''
	This is a lokahead step taken to check the safety of the system
	This method also calculates the reward for the RL actions before modification
	'''
	# Check for stability and safety on the nominal vehicle model
	def step_lookahead(self,action):
		
		done = False
		reward = 0
		stable_criteria = False
		safety_criteria = False

		#Extract throttle and steering information
		throttle = action[0]
		delta = action[1]
		# Normalizing the throttle and steering within the desired range
		delta = np.clip(delta,-0.001,0.001)

		v_previous = self.v_ego
		absolute_error_range = 0.2
		sigma = 5

		target_index, dx, dy, absolute_error = self.car.tracker.find_target_path_id(self.x_ego, self.y_ego, self.yaw_ego)
		yaw_error = self.car.tracker.calculate_yaw_term(target_index, self.yaw_ego)
		crosstrack_steering_error, crosstrack_error = self.car.tracker.calculate_crosstrack_term(self.v_ego, self.yaw_ego, dx, dy, absolute_error)

		# Call the dummy drive function, this does not affect the environment variables
		# Use local variables we don't want these to effect the environment
		x_ego, y_ego, yaw_ego, v_ego = self.car.drive_dummy(throttle,delta)

		# Check distance from front vehicle on nominal dynamics
		distance_fc = math.sqrt((x_ego - self.car_fc.x)**2 + (y_ego - self.car_fc.y)**2)

		#Check for stability of the lookahead state
		# 1. If velocity is better than the previous velocity but still within desired bounds
		# 2. If the absolute error is within bounds
		if v_ego > v_previous and v_ego < self.v_desired and absolute_error <=absolute_error_range:
			stable_criteria = True
		# Safety Criteria
		# 1. Check the distance with front vehicle
		if distance_fc > sigma:
			safety_criteria = True

		diff_target = np.sqrt((x_ego-self.x_target)**2 + (y_ego-self.y_target)**2)

		reward_speed_tracking = -abs(v_ego - self.v_desired)*0.1
		reward_orientation_tracking = crosstrack_error


		reward = reward + reward_speed_tracking+crosstrack_error
		#print(f'reward === {reward}')
		reward = reward - absolute_error
		if(round(absolute_error) == 0):
			reward = reward + 1+self.counter 

		#Reward for successful lane change
		if(round(x_ego,1) <= round(self.car_sc.x,1)):
			reward = reward + 1

		if(diff_target < 1):
			reward = reward + 100
			done = True

		if(self.counter > 400 or absolute_error>7):
			reward = reward + self.counter*0.5
			done = True
			reward = reward - 500

		obs = self.feature_scaling(np.hstack((x_ego,y_ego,v_ego, (self.v_desired-v_ego),diff_target)))

		info = [throttle,delta]
		#info = {}

		return stable_criteria,safety_criteria,reward,done,info

	
	


	'''
	This is the main step function which executes the step on the environment
	The action is a numpy 2d array having speed and steering values
	The reward is based on absolute error from path waypoints, steering error and difference with desired velocity
	'''
	def step(self,action):
		
		done = False
		self.counter = self.counter+1
		reward = 0

		#Extract throttle and steering information
		throttle = action[0]
		delta = action[1]
		# Normalizing the throttle and steering within the desired range
		#throttle = (np.clip(throttle, -1.0,1.0))  # -1.4..1.4
		delta = np.clip(delta,-0.001,0.001)

		self.distance_fc = math.sqrt((self.x_ego - self.car_fc.x)**2 + (self.y_ego - self.car_fc.y)**2)
		self.distance_sc = math.sqrt((self.x_ego - self.car_sc.x)**2 + (self.y_ego - self.car_sc.y)**2)

		# Incentive for moving forward
		if(self.distance_fc > 10):
			throttle = (np.clip(action[0], 0.0,1.0)+0.28)*1.1   # 0.3..1.35

		target_index, dx, dy, absolute_error = self.car.tracker.find_target_path_id(self.x_ego, self.y_ego, self.yaw_ego)
		yaw_error = self.car.tracker.calculate_yaw_term(target_index, self.yaw_ego)
		crosstrack_steering_error, crosstrack_error = self.car.tracker.calculate_crosstrack_term(self.v_ego, self.yaw_ego, dx, dy, absolute_error)


		self.x_ego, self.y_ego, self.yaw_ego, self.v_ego = self.car.drive(throttle,delta)

		#Prevent negative velocity
		if self.v_ego<0:
			self.v_ego = 0

		#Drive the other vehicles
		self.car_fc.drive(0.7,0.0)
		self.car_sc.drive(0.6,0.0)

		self.diff_target = np.sqrt((self.x_ego-self.x_target)**2 + (self.y_ego-self.y_target)**2)


		#Reward for speed tracking
		reward_speed_tracking = -abs(self.v_ego - self.v_desired)*0.1
		reward_orientation_tracking = crosstrack_error


		#reward = reward + reward_speed_tracking
		#print(f'reward === {reward}')
		reward = reward - absolute_error
		# Ensures that the car is moving
		if(round(absolute_error) == 0 and self.v_ego > 1):
			reward = reward + 0.5
		elif self.v_ego <= 1:
			reward = reward - 0.5
		#print(f'reward === {reward}')
		#Reward for following track
		#Reward for reaching first checkpoint

		#Reward for successful lane change
		if(round(self.x_ego,1) <= round(self.car_sc.x,1)):
			reward = reward + 1

		if(self.diff_target < 2):
			reward = reward + 500
			done = True

		if(self.counter > 550 or absolute_error>7):
			#reward = reward + self.counter*0.5
			done = True
			reward = reward - 500

		#Collision penalties
		if(self.distance_fc <=2.5):
			done = True
			reward = reward - 500
		

		obs = self.feature_scaling(np.hstack((self.x_ego,self.y_ego,self.v_ego, (self.v_desired-self.v_ego),self.diff_target)))

		info = [throttle,delta]
		#info = {}

		return np.array(obs, dtype=np.float32),reward,done,info


	def feature_scaling(self, state):
		"""
		Min-Max-Scaler: scale X' = (X-Xmin) / (Xmax-Xmin)
		:param state:
		:return: scaled state
		"""
		return (state - self.state_min) / (self.state_max - self.state_min)


	# This reset function brings environment to initial condition during render
	def reset_render(self):
		self.sim = Simulation()
		self.path = Path()

		#get the staring point of the path
		self.x_ego = self.path.x_path[0] #Ego vehicle's position
		self.y_ego = self.path.y_path[0]#+rand.randint(-5,5) #Ego vehicle's position
		
		self.car = Car(self.x_ego , self.y_ego , self.path.pyaw[0], self.path.px, self.path.py, self.path.pyaw, self.sim.dt)
		self.desc = Description(self.car.length, self.car.width, self.car.rear2wheel, self.car.wheel_dia, self.car.wheel_width, self.car.tread, self.car.L)

		self.x_fc = self.path.x_path_fc[0]
		self.y_fc = self.path.y_path_fc[0]

		self.car_fc = Car(self.x_fc , self.y_fc , self.path.pyaw_fc[0], self.path.px_fc, self.path.py_fc, self.path.pyaw_fc, self.sim.dt)
		self.desc_fc = Description(self.car_fc.length, self.car_fc.width, self.car_fc.rear2wheel, self.car_fc.wheel_dia, self.car_fc.wheel_width, self.car_fc.tread, self.car_fc.L)

		self.x_sc = self.path.x_path_sc[0]
		self.y_sc = self.path.y_path_sc[0]

		self.car_sc = Car(self.x_sc , self.y_sc , self.path.pyaw_sc[0], self.path.px_sc, self.path.py_sc, self.path.pyaw_sc, self.sim.dt)
		self.desc_sc = Description(self.car_sc.length, self.car_sc.width, self.car_sc.rear2wheel, self.car_sc.wheel_dia, self.car_sc.wheel_width, self.car_sc.tread, self.car_sc.L)


		self.v_ego = 0.0
		self.car.v = self.v_ego 

		self.car_fc.v = 0.0
		self.car_sc.v = 0.0


	def render(self,actionlist):
		global myvar
		global action_list
		action_list = actionlist
		myvar = 0
		self.reset_render()
		self.fig = plt.figure()
		self.ax = plt.axes()
		self.ax.set_aspect('equal')

		self.road = plt.Rectangle((30, 0), 50,50, color='gray', fill=False, linewidth=65)
		self.lane2 = plt.Rectangle((25, -5), 60,60, color='gold', fill=False, linestyle='--', linewidth=1.5)
		self.lane1 = plt.Rectangle((35, 5), 45,40, color='gold', fill=False, linestyle='--', linewidth=1.5)


		#road = plt.Circle((0, 0), 50, color='gray', fill=False, linewidth=30)
		self.ax.add_patch(self.road)
		self.ax.add_patch(self.lane1)
		self.ax.add_patch(self.lane2)
		self.ax.plot(self.path.px, self.path.py, '--', color='red')

		self.annotation = self.ax.annotate(f'{self.car.x:.1f}, {self.car.y:.1f}', xy=(self.car.x, self.car.y + 5), color='black', annotation_clip=False)
		self.target, = self.ax.plot([], [], '+r')

		self.outline, = self.ax.plot([], [], color=self.car.colour)
		self.fr, = self.ax.plot([], [], color=self.car.colour)
		self.rr, = self.ax.plot([], [], color=self.car.colour)
		self.fl, = self.ax.plot([], [], color=self.car.colour)
		self.rl, = self.ax.plot([], [], color=self.car.colour)
		self.rear_axle, = self.ax.plot(self.car.x, self.car.y, '+', color=self.car.colour, markersize=2)

		self.outline_fc, = self.ax.plot([], [], color='red')
		self.fr_fc, = self.ax.plot([], [], color='red')
		self.rr_fc, = self.ax.plot([], [], color='red')
		self.fl_fc, = self.ax.plot([], [], color='red')
		self.rl_fc, = self.ax.plot([], [], color='red')
		self.rear_axle_fc, = self.ax.plot(self.car_fc.x, self.car_fc.y, '+', color='red', markersize=2)

		self.outline_sc, = self.ax.plot([], [], color='blue')
		self.fr_sc, = self.ax.plot([], [], color='blue')
		self.rr_sc, = self.ax.plot([], [], color='blue')
		self.fl_sc, = self.ax.plot([], [], color='blue')
		self.rl_sc, = self.ax.plot([], [], color='blue')
		self.rear_axle_sc, = self.ax.plot(self.car_sc.x, self.car_sc.y, '+', color='blue', markersize=2)

		plt.grid()
		def animate(frame):
			global myvar
			global action_list
			frames=self.sim.frames
			# Camera tracks car
			# print(myvar)
			self.ax.set_xlim(self.car.x - self.sim.map_size, self.car.x + self.sim.map_size)
			self.ax.set_ylim(self.car.y - self.sim.map_size, self.car.y + self.sim.map_size)

			# Drive and draw car
			self.car.drive(action_list[myvar][0],action_list[myvar][1])
			outline_plot, fr_plot, rr_plot, fl_plot, rl_plot = self.desc.plot_car(self.car.x, self.car.y, self.car.yaw, self.car.delta)
			self.outline.set_data(outline_plot[0], outline_plot[1])
			self.fr.set_data(fr_plot[0], fr_plot[1])
			self.rr.set_data(rr_plot[0], rr_plot[1])
			self.fl.set_data(fl_plot[0], fl_plot[1])
			self.rl.set_data(rl_plot[0], rl_plot[1])
			self.rear_axle.set_data(self.car.x, self.car.y)

			self.car_fc.drive(0.7,0.0)
			outline_plot_fc, fr_plot_fc, rr_plot_fc, fl_plot_fc, rl_plot_fc = self.desc.plot_car(self.car_fc.x, self.car_fc.y, self.car_fc.yaw, self.car_fc.delta)
			self.outline_fc.set_data(outline_plot_fc[0], outline_plot_fc[1])
			self.fr_fc.set_data(fr_plot_fc[0], fr_plot_fc[1])
			self.rr_fc.set_data(rr_plot_fc[0], rr_plot_fc[1])
			self.fl_fc.set_data(fl_plot_fc[0], fl_plot_fc[1])
			self.rl_fc.set_data(rl_plot_fc[0], rl_plot_fc[1])
			self.rear_axle_fc.set_data(self.car_fc.x, self.car_fc.y)

			self.car_sc.drive(0.6,0.0)
			outline_plot_sc, fr_plot_sc, rr_plot_sc, fl_plot_sc, rl_plot_sc = self.desc.plot_car(self.car_sc.x, self.car_sc.y, self.car_sc.yaw, self.car_sc.delta)
			self.outline_sc.set_data(outline_plot_sc[0], outline_plot_sc[1])
			self.fr_sc.set_data(fr_plot_sc[0], fr_plot_sc[1])
			self.rr_sc.set_data(rr_plot_sc[0], rr_plot_sc[1])
			self.fl_sc.set_data(fl_plot_sc[0], fl_plot_sc[1])
			self.rl_sc.set_data(rl_plot_sc[0], rl_plot_sc[1])
			self.rear_axle_sc.set_data(self.car_sc.x, self.car_sc.y)

			# Show car's target
			#self.target.set_data(self.path.px[self.car.target_id], self.path.py[self.car.target_id])

			# Annotate car's coordinate above car
			self.annotation.set_text(f'{self.car.x:.1f}, {self.car.y:.1f}')
			self.annotation.set_position((self.car.x, self.car.y + 5))

			plt.title(f'{self.sim.dt*frame:.2f}s', loc='right')
			plt.xlabel(f'Speed: {self.car.v:.2f} m/s', loc='left')
			if myvar == len(action_list)-1:
				plt.close() 
				return None #exit()
			myvar = myvar+1

			return self.outline, self.fr, self.rr, self.fl, self.rl, self.rear_axle, self.target,

		_ = FuncAnimation(self.fig, animate, frames=self.sim.frames, interval=self.interval, repeat=self.sim.loop)
		# anim.save('animation.gif', writer='imagemagick', fps=50)
		plt.show()


if __name__=='__main__':
	global action_list
	done = False
	env = KinematicBicycleGymACC()
	obs = env.reset()
	action_list = []
	score = 0
	while not done:
		a_predicted_clf = clf_control(env.v_ego)
		distance = 10
		a_predicted_cbf = 0.0
		throttle = a_predicted_clf
		#Car in fc vehicles path
		if(round(env.x_ego,1) == round(env.car_fc.x,1)):
			distance = env.distance_fc
			vel_lead = env.car_fc.v
			try:
				a_predicted_cbf = cbf_control(env.v_ego,vel_lead,distance)
			except:
				a_predicted_cbf = -1.0
			print(f'distance before lc ======= {env.distance_fc}==speed===== {env.v_ego}')
			if(distance < 7):
				throttle = a_predicted_cbf
		delta, target_id, crosstrack_error = env.car.tracker.stanley_control(env.x_ego, env.y_ego, env.yaw_ego, env.v_ego, env.delta_ego)
		#print(f'crosstrack error ================ {crosstrack_error}')
		print(f'correct actions================ {[throttle,delta]}')
		obs,reward,done,_ = env.step([throttle,delta])
		score = score + reward
		action_list.append([throttle,delta])
	print(f'Final reward ========== {score} ======= {len(action_list)}')
	env.render(action_list)