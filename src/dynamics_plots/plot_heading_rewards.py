import numpy as np
import matplotlib.pyplot as plt

def calculate_heading_reward(heading_diff, sigma=np.pi/6):
    """
    Calculate heading reward using a Gaussian-like function.
    Args:
        heading_diff: Absolute difference between current heading and desired heading
        sigma: Standard deviation controlling how quickly reward drops off
    """
    reward = 2.0 * np.exp(-0.5 * (heading_diff/sigma)**2)
    
    # Add negative reward for very bad alignments
    if isinstance(heading_diff, np.ndarray):
        penalty = np.where(heading_diff > np.pi/2, -2.0, 0.0)
    else:
        penalty = -2.0 if heading_diff > np.pi/2 else 0.0
    
    return reward + penalty

def plot_heading_rewards():
    """Plot heading reward function for different parameters."""
    # Create array of heading differences
    heading_diffs = np.linspace(0, np.pi, 200)
    
    # Plot for different sigma values
    plt.figure(figsize=(12, 8))
    sigmas = [np.pi/8, np.pi/6, np.pi/4]  # Different spreads to compare
    
    for sigma in sigmas:
        rewards = calculate_heading_reward(heading_diffs, sigma)
        plt.plot(heading_diffs * 180/np.pi, rewards, 
                label=f'σ = {sigma*180/np.pi:.1f}°', linewidth=2)
    
    # Add reference lines
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=90, color='r', linestyle='--', alpha=0.3, 
                label='90° threshold')
    
    # Customize the plot
    plt.grid(True)
    plt.xlabel('Heading Difference (degrees)')
    plt.ylabel('Reward')
    plt.title('Heading Reward vs Heading Difference')
    plt.legend()
    
    # Add text box explaining the function
    text = (
        'Function:\n'
        'reward = 2.0 * exp(-0.5 * (heading_diff/σ)²)\n'
        'Additional -2.0 penalty when heading_diff > 90°'
    )
    plt.text(0.02, 0.98, text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    # Create a more detailed visualization around the critical points
    plt.figure(figsize=(12, 8))
    
    # Plot main reward curve
    sigma = np.pi/6  # Use default sigma
    rewards = calculate_heading_reward(heading_diffs, sigma)
    plt.plot(heading_diffs * 180/np.pi, rewards, 'b-', linewidth=2, label='Total Reward')
    
    # Plot components separately
    gaussian_part = 2.0 * np.exp(-0.5 * (heading_diffs/sigma)**2)
    penalty_part = np.where(heading_diffs > np.pi/2, -2.0, 0.0)
    
    plt.plot(heading_diffs * 180/np.pi, gaussian_part, 'g--', 
            label='Gaussian Component', alpha=0.7)
    plt.plot(heading_diffs * 180/np.pi, penalty_part, 'r--', 
            label='Penalty Component', alpha=0.7)
    
    # Add reference lines and annotations
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=90, color='r', linestyle='--', alpha=0.3)
    
    # Annotate key points
    plt.annotate('Maximum Reward\n(Perfect Alignment)', 
                xy=(0, 2), xytext=(20, 2.2),
                arrowprops=dict(facecolor='black', shrink=0.05))
    
    plt.annotate('90° Threshold\nPenalty Begins', 
                xy=(90, -0.5), xytext=(60, -1),
                arrowprops=dict(facecolor='black', shrink=0.05))
    
    # Customize the plot
    plt.grid(True)
    plt.xlabel('Heading Difference (degrees)')
    plt.ylabel('Reward')
    plt.title('Decomposition of Heading Reward Function (σ = 30°)')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_heading_rewards() 