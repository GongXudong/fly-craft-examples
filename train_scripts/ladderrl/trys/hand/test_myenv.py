
import gymnasium as gym
import register_myenv
import numpy as np 
import time
# env = gym.make("MyRotateZ-v0", render_mode="human")

env = gym.make("MyRotateZ-v0", min_angle_rad=-np.radians(45), max_angle_rad =np.radians(45) , render_mode="human")


num_episodes = 10  # 测试 100 个完整回合
total_success = 0

for e in range(num_episodes):
    obs, info = env.reset()
    done = False
    episode_success = False # 记录当前回合是否成功过
    
    while not done:
        action  =env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        time.sleep(1/60)

        if info.get("is_success",0) > 0:
            episode_success = True
            
        done = terminated or truncated

    if episode_success:
        total_success += 1
    
    print(f"Episode {e+1} finished. Success: {episode_success}")


env.close()

