import os
import yaml
import gym
from actor_critic import ActorCriticAgent
from train_agent import train_agent
import time

# list configs
configs = sorted(os.listdir('configs'))

start_time = time.time()
# for every config file
for config in configs:
    
    # load config
    with open(f"configs/{config}", 'r') as file:
        args = yaml.safe_load(file)
        
    # environment
    env = gym.make(args['env'])

    # environment properties
    dim_states = env.observation_space.shape[0]
    continuous_control = isinstance(env.action_space, gym.spaces.Box)
    dim_actions = env.action_space.shape[0] if continuous_control else env.action_space.n

    # instantiate agent
    actor_critic_agent = ActorCriticAgent(dim_states=dim_states,
                                          dim_actions=dim_actions,
                                          continuous_control=continuous_control,
                                          **args['agent'],
                                          )

    # train agent
    train_agent(env=env, 
                agent=actor_critic_agent, 
                **args['train'],
                )

execution_time = time.time() - start_time

print(f'{execution_time:.2f} seconds -- {(execution_time/60):.2f} minutes -- {(execution_time/3600):.2f} hours')
