from cvxopt import matrix, solvers
import matplotlib.pyplot as plt


#dictionary of defaults parameters
params = {'mass':1650,'g':9.81, 'f_0':0.1, 'f_1':5, 'f_2':0.25, 'v_ref':24 , 'v_0':13.89,'epsilon':10,'gamma':1,'c_a':0.3,'c_d':0.3,'p_sc':1e-5,'T_h':1.8}

def set_param(key_param,value_param):
	global params
	for i in range(len(key_param)):
		params[key_param[i]] = value_param[i]
	#print(params)

def clf_control(v_ego):
	global params
	#Calculate the rolling resistance
	F_r = params['f_0']+params['f_1']*v_ego+params['f_2']*(v_ego**2)
	#print(f'F_r ========= {F_r}')
	L_gv = (2/params['mass'])*(v_ego-params['v_ref'])
	#print(f'L_gv ========= {L_gv}')
	L_fv = (-2/params['mass'])*F_r*(v_ego-params['v_ref'])
	#print(f'L_fv ========= {L_fv}')
	solvers.options['show_progress'] = False

	Q = 2*matrix([[(1/(params['mass']**2)), 0], [0, params['p_sc']] ])
	p = -2*matrix([(F_r/(params['mass']**2)), 0.0])
	G = matrix([[L_gv,-1.0,1.0],[-1.0,0.0,0.0]])
	h = matrix([-5*(v_ego-params['v_ref'])**2-L_fv,4885.95,4885.95])
	sol = solvers.qp(Q, p, G, h)
	control_output = sol['x'][0]
	#print(f'control_output ========= {control_output}')
	a_ego = control_output/params['mass']
	return a_ego

def cbf_control(v_ego,v_lead,z):
	global params
	B = z-params['T_h']*v_ego-0.5*((v_ego-v_lead)**2/(params['c_d']*params['g']))
	#print(f'B ========= {B}')
	#Calculate the rolling resistance
	F_r = params['f_0']+params['f_1']*v_ego+params['f_2']*(v_ego**2)
	L_fb = (params['T_h'] + ((v_ego-v_lead)/params['c_d']*params['g']))*F_r/params['mass']+(v_lead-v_ego)
	L_gb = (params['T_h'] + ((v_ego-v_lead)/params['c_d']*params['g']))*1/params['mass']
	Q = 2*matrix([[(1/(params['mass']**2)), 0], [0, params['p_sc']] ])
	p = -2*matrix([(F_r/(params['mass']**2)), 0.0])
	G = matrix([[L_gb,-1.0,1.0],[0.0,0.0,0.0]])
	h = matrix([5*B+L_fb,4885.95,4885.95])
	sol = solvers.qp(Q, p, G, h)
	control_output = sol['x'][0]
	#print(f'control_output ========= {control_output}')
	a_ego = control_output/params['mass']
	#print(f'a_ego ========= {a_ego}')
	return a_ego

def plot(x,y):
	plt.plot(x, y)
	# naming the x axis
	plt.xlabel('time - s')
	# naming the y axis
	plt.ylabel('speed - m/s')
	
	# giving a title to my graph
	plt.title('CLF')
	
	# function to show the plot
	plt.show()


#if __name__=='__main__':
	#a_ego = clf_control(9)
	#print(f'a_ego ========= {a_ego}')
	#a_ego = cbf_control(20,0,100)
