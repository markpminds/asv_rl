import numpy as np
import matplotlib.pyplot as plt
from config import MAX_SPEED

def plot_desired_speed_curve(critical_distance=10.0, steepness=0.2, max_distance=40.0):
    """
    Plot the desired speed curve as a function of distance to goal.
    
    Args:
        critical_distance (float): Distance at which to start significant deceleration
        steepness (float): Controls how sharply the speed drops
        max_distance (float): Maximum distance to plot
    """
    # Create distance points
    distances = np.linspace(0, max_distance, 200)
    
    # Calculate desired speeds
    desired_speeds = MAX_SPEED * (1.0 / (1.0 + np.exp(-steepness * (distances - critical_distance))))
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(distances, desired_speeds, 'b-', linewidth=2, label='Desired Speed')
    
    # Add vertical line at critical distance
    plt.axvline(x=critical_distance, color='r', linestyle='--', 
                label=f'Critical Distance ({critical_distance}m)')
    
    # Add horizontal line at MAX_SPEED/2
    plt.axhline(y=MAX_SPEED/2, color='g', linestyle=':', 
                label=f'Half Max Speed ({MAX_SPEED/2}m/s)')
    
    # Customize the plot
    plt.grid(True)
    plt.xlabel('Distance to Goal (m)')
    plt.ylabel('Desired Speed (m/s)')
    plt.title('Desired Speed vs Distance to Goal')
    plt.legend()
    
    # Add text box with function parameters
    text = f'Function:\ndesired_speed = MAX_SPEED * (1/(1 + exp(-{steepness}*(d - {critical_distance}))))\n'
    text += f'MAX_SPEED = {MAX_SPEED}'
    plt.text(0.02, 0.98, text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Plot with different parameters for comparison
    plot_desired_speed_curve(critical_distance=10.0, steepness=0.3)