import matplotlib.pyplot as plt

def plot_performance_metrics(tr_steps_vec, avg_reward_vec, std_reward_vec, exp_id, n_points = 100):
    
    plot_steps = len(tr_steps_vec) // n_points if len(tr_steps_vec) > 1 else 1
    
    tr_steps_vec = tr_steps_vec[::plot_steps]
    avg_reward_vec = avg_reward_vec[::plot_steps]
    std_reward_vec = std_reward_vec[::plot_steps]
    
    plt.errorbar(tr_steps_vec, avg_reward_vec, yerr = std_reward_vec, marker = '.', color = 'C0')
    plt.xlabel('Training Iteration')
    plt.ylabel('Avg Reward')
    plt.grid('on')
    plt.savefig(f'figures/agg_experiments/{exp_id}.pdf')
    plt.close()