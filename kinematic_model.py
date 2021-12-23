#!/usr/bin/env python

import numpy as np
from math import copysign, sin

from libs.normalise_angle import normalise_angle

class KinematicBicycleModel():

	def __init__(self, L=1.0, max_steer=1.0, dt=0.05, c_r=0.0, c_a=0.0):
		"""
		2D Kinematic Bicycle Model

		At initialisation
		:param L:           (float) vehicle's wheelbase [m]
		:param max_steer:   (float) vehicle's steering limits [rad]
		:param dt:          (float) discrete time period [s]
		:param c_r:         (float) vehicle's coefficient of resistance 
		:param c_a:         (float) vehicle's aerodynamic coefficient

		At every time step
		:param x:           (float) vehicle's x-coordinate [m]
		:param y:           (float) vehicle's y-coordinate [m]
		:param yaw:         (float) vehicle's heading [rad]
		:param v:           (float) vehicle's velocity in the x-axis [m/s]
		:param throttle:    (float) vehicle's accleration [m/s^2]
		:param delta:       (float) vehicle's steering angle [rad]

		:return x:          (float) vehicle's x-coordinate [m]
		:return y:          (float) vehicle's y-coordinate [m]
		:return yaw:        (float) vehicle's heading [rad]
		:return v:          (float) vehicle's velocity in the x-axis [m/s]
		:return delta:      (float) vehicle's steering angle [rad]
		:return omega:      (float) vehicle's angular velocity [rad/s]
		"""

		self.dt = dt
		self.L = L
		self.max_steer = max_steer
		self.c_r = c_r
		self.c_a = c_a

	def kinematic_model(self, x, y, yaw, v, throttle, delta):

		params = {}
		sign = lambda v: copysign(1, v)  
		m = params.get('m', 1600.)              # vehicle mass, kg
		g = params.get('g', 9.8)                # gravitational constant, m/s^2
		Cr = params.get('Cr', 0.01)             # coefficient of rolling friction
		Cd = params.get('Cd', 0.32)             # drag coefficient
		rho = params.get('rho', 1.3)            # density of air, kg/m^3
		A = params.get('A', 2.4)                # car area, m^2
		alpha = params.get(
			'alpha', [40, 25, 16, 12, 10])      # gear ratio / wheel radius
		gear = 2
		omega_eng = alpha[int(gear)-1] * v      # engine angular speed
		F_eng = alpha[int(gear)-1] * self.motor_torque(omega_eng, params) * throttle
		#F_eng = np.clip(F_eng,150,200)
		# print(f'F_eng ======================= {F_eng}')
		Fg = m * g * np.sin(0)
		Fr  = m * g * Cr * sign(v)
		Fa = 1/2 * rho * Cd * A * abs(v) * v
		Fd = Fg + Fr + Fa

		# Compute the local velocity in the x-axis
		#f_load = v * (self.c_r + self.c_a * v)
		#print(f'f_load ======================= {Fd}')
		v += self.dt * ((F_eng - Fd)/m)

		# Compute the radius and angular velocity of the kinematic bicycle model
		delta = np.clip(delta, -self.max_steer, self.max_steer)

		# Compute the state change rate
		x_dot = v * np.cos(yaw)
		y_dot = v * np.sin(yaw)
		omega = v * np.tan(delta) / self.L

		# Compute the final state using the discrete time model
		x += x_dot * self.dt
		y += y_dot * self.dt
		yaw += omega * self.dt
		yaw = normalise_angle(yaw)
		
		return x, y, yaw, v, delta, omega

	def motor_torque(self, omega, params={}):
		# Set up the system parameters
		Tm = params.get('Tm', 190.)             # engine torque constant
		omega_m = params.get('omega_m', 420.)   # peak engine angular speed
		beta = params.get('beta', 0.4)          # peak engine rolloff

		return np.clip(Tm * (1 - beta * (omega/omega_m - 1)**2), 0, None)

def main():

	print("This script is not meant to be executable, and should be used as a library.")

if __name__ == "__main__":
	main()
