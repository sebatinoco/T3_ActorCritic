import os
import pandas as pd
from utils.plot_performance_metrics import plot_performance_metrics

def plot_experiment(config):

    files = os.listdir('metrics')

    exp_files = [file for file in files if config in file]

    exp_data = pd.DataFrame()

    for file in exp_files:
        
        file_data = pd.read_csv(f'metrics/{file}', sep = '\t')
        exp_data = pd.concat((exp_data, file_data), axis = 1)

    steps = exp_data['steps'].mean(axis = 1)
    avg_reward = exp_data['avg_reward'].mean(axis = 1)
    std_reward = exp_data['avg_reward'].std(axis = 1)

    plot_performance_metrics(steps, avg_reward, std_reward, config)