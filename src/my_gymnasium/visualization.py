import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arrow, Rectangle
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
from config import ENV_WIDTH, ENV_HEIGHT
import os
from pathlib import Path

# Get the src directory path
SRC_DIR = Path(__file__).parent

# Load images using relative paths
Corsair = mpimg.imread(os.path.join(SRC_DIR, "images", "Corsair.png"))
Containership = mpimg.imread(os.path.join(SRC_DIR, "images", "Containership.png"))


class ASVVisualizer:
    def __init__(self, width=ENV_WIDTH, height=ENV_HEIGHT):
        self.width = width
        self.height = height
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.setup_plot()
        self.path = []  # Store vessel's path
        
    def setup_plot(self):
        """Setup the plot area"""
        self.ax.set_xlim(-self.width/2, self.width/2)
        self.ax.set_ylim(-self.height/2, self.height/2)
        self.ax.grid(True)
        self.ax.set_aspect('equal')
        self.ax.set_title('ASV Environment')
        
    def draw_vessel(self, x, y, heading, size=5):
        """Draw the vessel as an arrow"""
        # Store position for path
        self.path.append((x, y))
        
        # Clear previous drawings
        self.ax.clear()
        self.setup_plot()
        
        # Draw path
        if len(self.path) > 1:
            path = np.array(self.path)
            self.ax.plot(path[:, 0], path[:, 1], 'b--', alpha=0.5)  # Draw path as dashed line
        
        # Create arrow for vessel
        dx = size * np.cos(heading)
        dy = size * np.sin(heading)
        arrow = Arrow(x, y, dx, dy, width=size/2, color='blue')
        self.ax.add_patch(arrow)
        
        # Draw vessel position circle
        imagebox = OffsetImage(Corsair, zoom=0.1)
        ab = AnnotationBbox(imagebox, (x, y), frameon=False)
        self.ax.add_artist(ab)
        
    def draw_obstacles(self, obstacles):
        """Draw obstacles as circles"""
        for obs in obstacles:
            x, y, radius = obs
            obstacle = Circle((x, y), radius, 
                            color='orange',      # Edge color
                            fill=False,       # Make circle hollow
                            linewidth=1,      # Make the line more visible
                            alpha=1.0)        # Full opacity for the edge
            self.ax.add_patch(obstacle)
            imagebox = OffsetImage(Containership, zoom=0.07)
            ab = AnnotationBbox(imagebox, (x, y), frameon=False)
            self.ax.add_artist(ab)
            
            
    def draw_goal(self, x, y, radius=3):
        """Draw the goal position"""
        goal = Circle((x, y), radius, color='green', alpha=0.3)
        self.ax.add_patch(goal)
        
    def draw_threat_zones(self, threat_zones):
        """Draw threat zones as hollow red rectangles"""
        for zone in threat_zones:
            x, y, width, height = zone
            # Rectangle expects bottom-left corner, so adjust x,y which are center points
            rect = Rectangle(
                (x - width/2, y - height/2),  # bottom-left corner
                width, height,                 # width, height
                color='red',                   # edge color
                fill=False,                    # make rectangle hollow
                linewidth=1,                   # line thickness
                linestyle='--',                # dashed line
                alpha=0.7                      # slight transparency
            )
            self.ax.add_patch(rect)
        
    def render(self):
        """Update the visualization"""
        plt.draw()
        plt.pause(0.01)  # Small pause to update the plot
 