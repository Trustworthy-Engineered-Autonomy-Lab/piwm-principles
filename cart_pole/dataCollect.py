import gym
import numpy as np
import os
from PIL import Image
if __name__ == '__main__':

    save_dir = "./cartpole_data"
    os.makedirs(save_dir, exist_ok=True)

    noise_levels = [0.025, 0.05, 0.10]
    position_ref = 4.8*2
    angle_ref = 0.84

    def collect_and_save(run_id, max_steps=500):
        env = gym.make("CartPole-v1")
        obs = env.reset()[0]

        imgs = []
        acts = []
        states = []

        for _ in range(max_steps):
            # img = env.render()
            # env.render()
            img = env.render(mode="rgb_array") 
            action = env.action_space.sample()

            imgs.append(img)
            acts.append(action)
            states.append(obs)

            obs, _, terminated, truncated = env.step(action)
            if terminated or truncated:
                break

        env.close()

        imgs = np.array(imgs[1:])
        acts = np.array(acts[1:])
        states = np.array(states[1:])

        noisy_versions = {}
        for level in noise_levels:
            noise = np.zeros_like(states)
            noise[:, 0] = np.random.normal(0, level * position_ref, size=states.shape[0])
            noise[:, 2] = np.random.normal(0, level * angle_ref, size=states.shape[0])
            noisy_states = states + noise
            noisy_versions[f"noisy_states_{int(level*100)}"] = noisy_states

        save_path = os.path.join(save_dir, f"{run_id}.npz")
        np.savez_compressed(save_path, imgs=imgs, acts=acts, states=states, **noisy_versions)
        print(f"Saved to {save_path}")

    for i in range(5000):
        print(i)
        collect_and_save(i)
