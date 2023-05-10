import argparse
from distutils.util import strtobool

def parse_args():
    
    parser = argparse.ArgumentParser()
    
    # arguments
    parser.add_argument('--env', type = str, default = 'Pendulum-v1', help = 'environment to run the experiment')
    parser.add_argument('--training_iterations', type = int, default = 100, help = 'training iterations of the experiment')
    parser.add_argument('--actor_lr', type = float, default = 1e-3, help = 'learning rate of the algorithm')
    parser.add_argument('--critic_lr', type = float, default = 1e-2, help = 'learning rate of the algorithm')
    parser.add_argument('--gamma', type = float, default = .99, help = 'discount factor of the algorithm')
    parser.add_argument('--batch_size', type = int, default = 500, help = 'batch size of the algorithm')
    parser.add_argument('--nb_critic_updates', type = int, default = 100, help = 'number of updates performed by critic')
    parser.add_argument('--use_baseline', type=lambda x: bool(strtobool(x)), default = False, help = 'enable or disable baseline') 
    parser.add_argument('--reward_to_go', type=lambda x: bool(strtobool(x)), default = False, help = 'enable or disable reward to go')
    parser.add_argument('--exp_name', type = str, default = 'experiment', help = 'name of the experiment, used to store results')
    
    # consolidate args
    args = parser.parse_args()
    args = vars(args)
    
    return args