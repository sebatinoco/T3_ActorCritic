import argparse
from distutils.util import strtobool

def run_args():
    
    parser = argparse.ArgumentParser()
    
    # arguments
    parser.add_argument('--env', nargs = '+', default = [], help = 'environments to run the experiments')
    parser.add_argument('--filter_config', nargs = '+', default = [], help = 'specific config to run the experiments')
    parser.add_argument('--gpu', default = 0, type = int, help = 'gpu to run the experiments')
    parser.add_argument('--n_trials', default = 3, type = int, help = 'number of trials per experiment')
    
    
    # consolidate args
    args = parser.parse_args()
    args = vars(args)
    
    return args