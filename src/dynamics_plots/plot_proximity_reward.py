import numpy as np
import matplotlib.pyplot as plt
from my_gymnasium.reward_functions import proximity_reward

def plot_proximity_rewards():
    # Create array of distances from 0 to 40 with fine granularity
    distances = np.linspace(0, 40, 1000)
    
    # Calculate rewards for each distance
    rewards = [proximity_reward(d) for d in distances]
    
    plt.figure(figsize=(12, 6))
    
    # Plot the reward curve
    plt.plot(distances, rewards, 'b-', linewidth=2.5, label='Proximity Reward')
    
    # Add vertical line at threshold
    plt.axvline(x=30.0, color='r', linestyle='--', alpha=0.5, 
                label='Reward Threshold (30.0)')
    
    # Add horizontal line at y=0
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.2)
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Customize the plot
    plt.title('Proximity Reward vs Distance to Goal', fontsize=14, pad=20)
    plt.xlabel('Distance to Goal', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    
    # Add legend
    plt.legend(fontsize=10, loc='upper right')
    
    # Add annotations for key points
    plt.annotate(f'Maximum Reward\n(20.0)', 
                xy=(0, 20.0), 
                xytext=(5, 18),
                ha='left',
                arrowprops=dict(facecolor='black', shrink=0.05))
    
    plt.annotate(f'Exponential Decay\n(reward ≈ 7.4)', 
                xy=(10, proximity_reward(10)), 
                xytext=(12, 10),
                ha='left',
                arrowprops=dict(facecolor='black', shrink=0.05))
    
    plt.annotate(f'Reward ≈ 0\n(at threshold)', 
                xy=(30, proximity_reward(30)), 
                xytext=(25, 5),
                ha='right',
                arrowprops=dict(facecolor='black', shrink=0.05))
    
    # Set axis limits
    plt.ylim(-1, 22)
    plt.xlim(0, 40)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_proximity_rewards() 