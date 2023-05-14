import gym
from actor_critic import ActorCriticAgent
from train_agent import perform_single_rollout, sample_rollouts

# cambiar esta función para testear las dimensiones de los rollouts, quizas hacer el testing en esta funcion?
    
def test_methods(n_trials = 10):
        
    environments = ['CartPole-v1', 'Pendulum-v1']
    for environment in environments:
        
        # test "n_trials" times the shape 
        for _ in range(n_trials):
        
            env = gym.make(environment) # initiate environment
            dim_states = env.observation_space.shape[0] # compute state space
            continuous_control = isinstance(env.action_space, gym.spaces.Box) # True if action space is continuous, False if not 
            dim_actions = env.action_space.shape[0] if continuous_control else env.action_space.n # compute action space
            
            # initiate agent
            agent = ActorCriticAgent(
                dim_states = dim_states,
                dim_actions = dim_actions,
                actor_lr = 1e-3,
                critic_lr = 1e-3,
                gamma = .99,
                continuous_control = continuous_control
            )

            # test for perform_single_rollout
            single_rollout = perform_single_rollout(env = env, agent = agent)
            
            # test for sample_rollouts
            sampled_rollouts = sample_rollouts(env = env, agent = agent, training_iter = 1, min_batch_steps = 250)