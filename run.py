import os
import yaml
import gym
from actor_critic import ActorCriticAgent
from train_agent import train_agent
import time
from utils.run_args import run_args

# run args
r_args = run_args()

filter_env = r_args['env']
filter_config = r_args['filter_config']
gpu = r_args['gpu']
n_trials = r_args['n_trials']

# list configs
configs = sorted(os.listdir('configs'))

# filter configs if specified
if filter_env or filter_config:
    
    env_configs = [config for config in configs if len(set(filter_env) & set(config.split('_'))) > 0] # filter by environment
    filtered_configs = [config for config in configs if config in filter_config] # filter by config
    
    final_configs = set(env_configs + filtered_configs) # filtered configs
    configs = [config for config in configs if config in final_configs] # filter configs

print('Running experiments on the following configs: ', configs)

for trial in range(1, n_trials + 1):
    start_time = time.time()
    # for every config file
    for config in configs:
        
        # load config
        with open(f"configs/{config}", 'r') as file:
            args = yaml.safe_load(file)
        
        # experiment name
        exp_name = f"{args['env'][:-3]}_{args['exp_name']}_{trial}"
        
        # environment
        env = gym.make(args['env'])

        # environment properties
        dim_states = env.observation_space.shape[0]
        continuous_control = isinstance(env.action_space, gym.spaces.Box)
        dim_actions = env.action_space.shape[0] if continuous_control else env.action_space.n

        # instantiate agent
        actor_critic_agent = ActorCriticAgent(dim_states = dim_states,
                                            dim_actions = dim_actions,
                                            continuous_control = continuous_control,
                                            gpu = gpu,
                                            **args['agent'],
                                            )

        # train agent
        train_agent(env = env, 
                    agent = actor_critic_agent, 
                    exp_name = exp_name,
                    **args['train'],
                    )

    execution_time = time.time() - start_time

    print(f'{execution_time:.2f} seconds -- {(execution_time/60):.2f} minutes -- {(execution_time/3600):.2f} hours')
