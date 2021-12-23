# afe and Stable RL (S2RL) Driving Policies Using Control Barrier and Control Lyapunov Functions

# System Requirements

The code has been tested in systems with the following OS

- Ubuntu 20.04.2 LTS

## Installation

1. Setup conda environment

```
$ conda create -n env_name python=3.8.5
$ conda activate env_name
```
2. Clone the repository to an appropriate folder
3. Install requirements

```
$ pip install -r requirements.txt
$ pip install -e .
```


## Usage

All the trained policies are avialable in the policies folder

```


The pre-trained policies are available in the Policies folder. Following environement files are available for training

1) KinematicBicycleGymCutACC - Straight Drive with limited steering
2) KinematicBicycleGymTurn - Training Left/Right Turning agents
3) KinematicBicycleGymLane - Training Left/Right Lane change
4) KinematicBicycleGymCutIn - Training Left/Right Lane change

The main program takes the following command line arguments

1) --env : environment name (default is KinematicBicycleGymCutACC)
2) --actor : filepath to the actor network (default is Policies/ppo_actorPendulum-v0.pth)
3) --critic : filepath to the critic network (default is Policies/ppo_criticPendulum-v0.pth)
4) --failuretraj : The filepath to the failure trajectory file (default is Failure_Trajectories/failure_trajectory_pendulum.data)
5) --isdiscrete : True if environment is discrete (default False)
6) --oldactor : filepath to the original actor network (default is Policies/ppo_actorPendulum-v0.pth)
7) --oldcritic : filepath to the original critic network (default is Policies/ppo_criticPendulum-v0.pth)
8) --subactor : filepath to the subpolicy actor network (default is Policies/ppo_actor_subpolicyPendulum-v0.pth)
9) --subcritic : filepath to the subpolicy critic network (default is Policies/ppo_critic_subpolicyPendulum-v0.pth)
10) --newactor : filepath to the updated actor network (default is Policies/ppo_actor_updatedPendulum-v0.pth)

The hyperparameters can be changed in the hyperparameters.yml file


Note : Change the default arguments inside the main.py file otherwise the command line may become too long


### Testing

To test a trained model run:

```
$ python main.py --test
```

Press ctr+c to end testing

### Generating Failure trajectories for a specific environment

Failure trajectories uncovered with our tests are available in Failure_Trajectories Folder

Each environment has a seperate Bayesian Optimization file. Run the Bayesian Optimization correspondig to the environment
We use GpyOpt Library for Bayesian Optimization. As per (https://github.com/SheffieldML/GPyOpt/issues/337) GpyOpt has stochastic evaluations even when the seed is fixed.
This may lead to identification of a different number failure trajectories (higher or lower) than the mean number of trajectories reported in the paper.

For example to generate failure trajectories for the Pendulum environment run:

```
$ python PendulumBO.py
```

The failure trajectories will be written in the corresponding data files in the same folder

### Displaying Failure trajectories

To display failure trajectories:

```
$ python main.py --display
```
Mention the actor policy and the failure trajectory file in arguments or in the main.py file

Change the actor_model argument for observing the behaviour of sub-policy and updated policy on the failure trajectories


### Training the sub-policy

```
$ python main.py --subtrain
```

Mention the failure trajectory file in arguments or in the main.py file

### Update the original Policy to new Policy via gradient based updates

```
$ python main.py --correct
```
The correct method takes the actor and critic networks of the old policy and the subpolicy as an argument

default function parameters are 
1) --env : environment name (default is Pendulum-v0)
2) --oldactor : filepath to the original actor network (default is Policies/ppo_actorPendulum-v0.pth)
3) --oldcritic : filepath to the original critic network (default is Policies/ppo_criticPendulum-v0.pth)
4) --subactor : filepath to the subpolicy actor network (default is Policies/ppo_actor_subpolicyPendulum-v0.pth)
5) --subcritic : filepath to the subpolicy critic network (default is Policies/ppo_critic_subpolicyPendulum-v0.pth)
6) --failuretraj : The filepath to the failure trajectory path (default is FFailure_Trajectories/failure_trajectory_pendulum.data)
7) --isdiscrete : True if environment is discrete (default False)

### Calculate the distance between the original policy and the updated policy

```
$ python main.py --distance
```
default function parameters are:
1) --oldactor : filepath to the original actor network (default is Policies/ppo_actorPendulum-v0.pth)
2) --newactor : filepath to the updated actor network (default is Policies/ppo_actor_updatedPendulum-v0.pth)
3) --env : environment name (default is LunarLanderContinuous-v2)
4) --isdiscrete : True if environment is discrete (default False)


### Display the plots and heatmaps

```
$ tensorboard --logdir=bestruns
```
```
$ plot_heatmap_pendulum.py
```

The heatmaps are stored inside the img folder

### Training a policy from scratch

To train a model run:

```
$ python main.py --train
```
The hyperparameters can be changed in the hyperparameters.yml file
