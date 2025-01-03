import numpy as np
from environment import ASVEnvironment
import time
import matplotlib.pyplot as plt

def test_straight_movement():
    """Test moving straight ahead"""
    env = ASVEnvironment()
    obs, _ = env.reset()
    
    print("Testing straight movement...")
    for _ in range(20):
        action = np.array([1.0, 0.0])  # Full throttle, no rudder
        obs, reward, done, _, _ = env.step(action)
        env.render()
        print(f"Position: ({obs['vessel_state'][0]:.2f}, {obs['vessel_state'][1]:.2f}), "
              f"Speed: {obs['vessel_state'][3]:.2f}, "
              f"Heading: {obs['vessel_state'][2]:.2f}")
        time.sleep(0.1)
        break

def test_turning():
    """Test turning behavior"""
    env = ASVEnvironment()
    obs, _ = env.reset()
    
    print("\nTesting turning movement...")
    for _ in range(5):
        action = np.array([0.5, np.pi/4])  # Half throttle, full right rudder
        obs, reward, done, _, _ = env.step(action)
        env.render()
        print(f"Position: ({obs['vessel_state'][0]:.2f}, {obs['vessel_state'][1]:.2f}), "
              f"Angular Velocity: {obs['vessel_state'][4]:.2f}, "
              f"Heading: {obs['vessel_state'][2]:.2f}")
        time.sleep(0.1)
        if done:
            break

def test_random_actions():
    """Test random actions"""
    env = ASVEnvironment()
    obs, _ = env.reset()
    
    print("\nTesting random actions...")
    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, done, _, _ = env.step(action)
        env.render()
        print(f"Action: {action}")
        print(f"Position: ({obs['vessel_state'][0]:.2f}, {obs['vessel_state'][1]:.2f}), "
              f"Speed: {obs['vessel_state'][3]:.2f}, "
              f"Heading: {obs['vessel_state'][2]:.2f}")
        time.sleep(0.1)
        if done:
            break

if __name__ == "__main__":
    # Run tests
    test_straight_movement()
    #test_turning()
    #test_random_actions()
    
    plt.show(block=True)  # This will keep all figures open 