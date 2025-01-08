import numpy as np
import matplotlib.pyplot as plt
from config import MAX_SPEED, TURN_RATE_FACTOR

def plot_turning_dynamics():
    """
    Plot angular acceleration as a function of rudder angle for different speeds.
    Shows how turning effectiveness decreases with speed.
    """
    # Create arrays for rudder angles and speeds
    rudder_angles = np.linspace(-np.pi/4, np.pi/4, 100)  # Full range of rudder angles
    speeds = [0.0, MAX_SPEED/4, MAX_SPEED/2, 3*MAX_SPEED/4, MAX_SPEED]  # Different speeds to plot
    
    plt.figure(figsize=(12, 8))
    
    # Plot angular acceleration for each speed
    for speed in speeds:
        turn_effectiveness = 1.0 / (1.0 + (speed / MAX_SPEED))
        angular_accels = TURN_RATE_FACTOR * rudder_angles * turn_effectiveness
        
        plt.plot(rudder_angles * 180/np.pi, angular_accels, 
                label=f'Speed = {speed:.1f} m/s', linewidth=2)
    
    # Add reference lines
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    # Customize the plot
    plt.grid(True)
    plt.xlabel('Rudder Angle (degrees)')
    plt.ylabel('Angular Acceleration (rad/s²)')
    plt.title('Angular Acceleration vs Rudder Angle at Different Speeds')
    plt.legend()
    
    # Add text box explaining the model
    text = (
        f'Model:\n'
        f'turn_effectiveness = 1/(1 + speed/MAX_SPEED)\n'
        f'angular_accel = {TURN_RATE_FACTOR:.2f} * rudder * turn_effectiveness\n'
        f'MAX_SPEED = {MAX_SPEED:.1f} m/s'
    )
    plt.text(0.02, 0.98, text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    # Create a 3D surface plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create meshgrid for 3D plot
    rudder_2d = np.linspace(-np.pi/4, np.pi/4, 50)
    speed_2d = np.linspace(0, MAX_SPEED, 50)
    R, S = np.meshgrid(rudder_2d, speed_2d)
    
    # Calculate angular acceleration for each point
    turn_effectiveness_2d = 1.0 / (1.0 + (S / MAX_SPEED))
    angular_accels_2d = TURN_RATE_FACTOR * R * turn_effectiveness_2d
    
    # Create the surface plot
    surface = ax.plot_surface(R * 180/np.pi, S, angular_accels_2d, 
                            cmap='viridis', alpha=0.8)
    
    # Customize the 3D plot
    ax.set_xlabel('Rudder Angle (degrees)')
    ax.set_ylabel('Speed (m/s)')
    ax.set_zlabel('Angular Acceleration (rad/s²)')
    ax.set_title('Angular Acceleration as a Function of\nRudder Angle and Speed')
    
    # Add colorbar
    fig.colorbar(surface, ax=ax, label='Angular Acceleration (rad/s²)')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_turning_dynamics() 