import pandas as pd
import matplotlib.pyplot as plt

def plot_results(csv_file):
    # Read CSV data
    df = pd.read_csv(csv_file)
    # Convert columns to numpy arrays to avoid multi-dimensional indexing issues
    timesteps = df['timesteps'].to_numpy()
    avg_return = df['avg_episode_return'].to_numpy()
    # Plot average episodic return vs. timesteps
    plt.figure()
    plt.plot(timesteps, avg_return, label='Avg Episodic Return')
    plt.xlabel("Training Timesteps")
    plt.ylabel("Average Episodic Return")
    plt.title("PPO: Pendulum-v1, Seed-102 -- Average Episodic Return Over Training Environment Steps")
    plt.legend()
    plt.show()

# Example usage: plot the CSV from one run
plot_results("/Users/admin/Desktop/Sem4/DeepRL/assignment_4/results/assignment4_Pendulum-v1_PPO_seed102_20250411_213955.csv")
