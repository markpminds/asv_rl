import numpy as np
import matplotlib.pyplot as plt
from my_gymnasium.reward_functions import speed_reward

def plot_speed_rewards():
    # Create array of speed ratios from 0 to 2.0 with fine granularity
    speed_ratios = np.linspace(0, 2.0, 1000)
    
    # Plot for different fixed distances
    distances = [5.0, 10.0, 15.0, 20.0, 30.0]
    colors = ['red', 'green', 'blue', 'purple', 'orange']
    linestyles = ['-', '-', '-', '-', '-']
    
    plt.figure(figsize=(12, 8))
    
    # Plot rewards for each distance
    for distance, color, style in zip(distances, colors, linestyles):
        rewards = [speed_reward(ratio, distance) for ratio in speed_ratios]
        plt.plot(speed_ratios, rewards, color=color, linestyle=style, linewidth=2.5, 
                label=f'Distance = {distance}m')
    
    # Add vertical lines at key speed ratios with different style
    plt.axvline(x=0.3, color='gray', linestyle=':', alpha=0.5, label='Speed Ratio = 0.3')
    plt.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5, label='Speed Ratio = 0.5')
    plt.axvline(x=0.75, color='gray', linestyle=':', alpha=0.5, label='Speed Ratio = 0.75')
    plt.axvline(x=1.0, color='gray', linestyle=':', alpha=0.5, label='Speed Ratio = 1.0')
    
    # Add horizontal line at y=0
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.2)
    
    # Customize the plot
    plt.grid(True, alpha=0.3)
    plt.title('Speed Reward vs Speed Ratio at Different Distances', fontsize=14, pad=20)
    plt.xlabel('Speed Ratio', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    
    # Adjust legend
    plt.legend(fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5),
              frameon=True, facecolor='white', framealpha=1)
    
    # Add annotations for key regions
    plt.annotate('Close Range\nOptimal Speed', 
                xy=(0.75, 5), 
                xytext=(0.75, 7),
                ha='center',
                arrowprops=dict(facecolor='black', shrink=0.05))
    
    plt.annotate('Long Range\nHigh Speed', 
                xy=(1.0, 4), 
                xytext=(1.2, 6),
                ha='center',
                arrowprops=dict(facecolor='black', shrink=0.05))
    
    # Adjust plot limits to show full range of rewards
    plt.ylim(-25, 8)
    plt.xlim(0, 2.0)
    
    # Ensure no clipping of labels
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_speed_rewards() 