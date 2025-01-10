import numpy as np
import matplotlib.pyplot as plt
from my_gymnasium.reward_functions import heading_reward


def plot_reward_function():
    # Create array of heading differences from 0 to pi
    heading_diffs = np.linspace(0, np.pi, 1000)
    rewards = np.array([heading_reward(diff) for diff in heading_diffs])
    
    # Convert radians to degrees for better readability
    heading_diffs_deg = np.degrees(heading_diffs)
    
    plt.figure(figsize=(12, 8))
    plt.plot(heading_diffs_deg, rewards, 'b-', linewidth=2)
    
    # Add vertical lines at key points
    plt.axvline(x=30, color='r', linestyle='--', alpha=0.5, label='π/6 (30°)')
    plt.axvline(x=45, color='g', linestyle='--', alpha=0.5, label='π/4 (45°)')
    plt.axvline(x=90, color='m', linestyle='--', alpha=0.5, label='π/2 (90°)')
    
    # Add horizontal line at y=0
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.2)
    
    # Customize the plot
    plt.grid(True, alpha=0.3)
    plt.title('Heading Reward Function (Gaussian)', fontsize=14, pad=20)
    plt.xlabel('Heading Difference (degrees)', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.legend(fontsize=10)
    
    # Add annotations for key points
    plt.annotate(f'Max Reward\n({heading_reward(0):.1f})', 
                xy=(0, heading_reward(0)), 
                xytext=(10, 11),
                arrowprops=dict(facecolor='black', shrink=0.05))
    
    plt.annotate(f'At 30°\n({heading_reward(np.pi/6):.1f})', 
                xy=(30, heading_reward(np.pi/6)), 
                xytext=(40, 5),
                arrowprops=dict(facecolor='black', shrink=0.05))
    
    plt.annotate(f'At 90°\n({heading_reward(np.pi/2):.1f})', 
                xy=(90, heading_reward(np.pi/2)), 
                xytext=(100, 0.5),
                arrowprops=dict(facecolor='black', shrink=0.05))
    
    plt.annotate(f'Min Reward\n({heading_reward(np.pi):.1f})', 
                xy=(180, heading_reward(np.pi)), 
                xytext=(160, -8),
                arrowprops=dict(facecolor='black', shrink=0.05))
    
    # Set axis limits to show full range
    plt.ylim(-12, 12)
    plt.xlim(-5, 185)
    
    # Ensure no clipping of labels
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_reward_function() 