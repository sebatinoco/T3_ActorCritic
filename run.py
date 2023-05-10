import os
import yaml
import gym
from actor_critic import ActorCriticAgent
from train_agent import train_agent

for config in os.listdir('configs'):
    
    with open(f"configs/{config}", 'r') as file:
        args = yaml.safe_load(file)
        
    env = gym.make(args['env'])

    dim_states = env.observation_space.shape[0]
    continuous_control = isinstance(env.action_space, gym.spaces.Box)
    dim_actions = env.action_space.shape[0] if continuous_control else env.action_space.n

    actor_critic_agent = ActorCriticAgent(dim_states=dim_states,
                                          dim_actions=dim_actions,
                                          continuous_control=continuous_control,
                                          **args['agent'],
                                          )

    train_agent(env=env, 
                agent=actor_critic_agent, 
                **args['train'],
                )
