import numpy as np
import matplotlib.pyplot as plt
from my_gymnasium.reward_functions import time_penalty
from config import TIME_PENALTY_THRESHOLD

def plot_time_penalties():
    # Create array of steps from 0 to 1000
    steps = np.linspace(0, 1000, 1000)
    
    # Calculate penalties for each step
    penalties = [time_penalty(step) for step in steps]
    
    plt.figure(figsize=(12, 6))
    
    # Plot the penalty curve
    plt.plot(steps, penalties, 'r-', linewidth=2.5, label='Time Penalty')
    
    # Add vertical line at threshold
    plt.axvline(x=TIME_PENALTY_THRESHOLD, color='b', linestyle='--', alpha=0.5, 
                label=f'Penalty Threshold ({TIME_PENALTY_THRESHOLD} steps)')
    
    # Add horizontal line at y=0
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.2)
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Customize the plot
    plt.title('Time Penalty vs Steps', fontsize=14, pad=20)
    plt.xlabel('Steps', fontsize=12)
    plt.ylabel('Penalty', fontsize=12)
    
    # Add legend
    plt.legend(fontsize=10, loc='lower left')
    
    # Add annotations for key points
    plt.annotate(f'Initial Penalty\n(-0.01)', 
                xy=(200, -0.01), 
                xytext=(150, -0.5),
                ha='right',
                arrowprops=dict(facecolor='black', shrink=0.05))
    
    plt.annotate(f'Exponential Growth\nBegins', 
                xy=(400, time_penalty(400)), 
                xytext=(450, -1),
                ha='left',
                arrowprops=dict(facecolor='black', shrink=0.05))
    
    plt.annotate(f'Maximum Penalty\n(≈ {time_penalty(1000):.1f})', 
                xy=(1000, time_penalty(1000)), 
                xytext=(800, time_penalty(1000)-1),
                ha='center',
                arrowprops=dict(facecolor='black', shrink=0.05))
    
    # Set axis limits
    plt.ylim(time_penalty(1000)-1, 0.5)  # Set y-axis to show full range of penalties
    plt.xlim(0, 1050)  # Add some padding to x-axis
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_time_penalties() 