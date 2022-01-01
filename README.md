# Safe and Stable RL (S2RL) Driving Policies Using Control Barrier and Control Lyapunov Functions

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
2) --actor : filepath to the actor network (default is ACCS2RLPPO/ppo_actorKinematicBicycleGymACC.pth)
3) --critic : filepath to the critic network (default is ACCS2RLPPO/ppo_criticKinematicBicycleGymACC.pth)

The hyperparameters can be changed in the hyperparameters.yml file


Note : Change the default arguments inside the main.py file otherwise the command line may become too long


### Testing

To test a trained model run:

```
$ python main.py --test
```

### Training the policy for a env

```
$ python main.py --train
```
Specify the appropriate environment for training

The GIF demonstrations are stored inside the demonstrations folder
