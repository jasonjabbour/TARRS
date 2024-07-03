import torch
import time
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from safe_ptp.src.env.spanning_tree_env import SpanningTreeEnv
from safe_ptp.src.alg.custom_gcn_policy import CustomGNNActorCriticPolicy


# Example usage
if __name__ == "__main__":
    from stable_baselines3.common.utils import get_schedule_fn
    learning_rate = 0.001
    lr_schedule = get_schedule_fn(learning_rate)

    # Initialize the environment
    env = SpanningTreeEnv(min_nodes=5, 
                          max_nodes=5, 
                          min_redundancy=3, 
                          max_redundancy=4, 
                          min_attacked_nodes=1, 
                          max_attacked_nodes=2,
                          start_difficulty_level=5,
                          final_difficulty_level=5,
                          num_timestep_cooldown=2, 
                          show_weight_labels=False, 
                          render_mode=True, 
                          max_ep_steps=3, 
                          node_size=250, 
                          performance_threshold=-1)


    policy = CustomGNNActorCriticPolicy(
        env.observation_space,
        env.action_space,
        lr_schedule,
        features_dim=256
    )

    obs = env.reset()

    for _ in range(100):
        action_logits, value, _ = policy(obs)
        # Print the shape and optionally the logits
        print("Shape of action_logits:", action_logits.shape)
        print("Action Logits", action_logits)
        action = (action_logits > 0.5).long().numpy()
        print("Action", action[0])
        obs, reward, done, truncated, info = env.step(action[0])
        print(f"Action: {action}, Reward: {reward}")
        # if done:
        #     obs = env.reset()


# if __name__ == "__main__":


#        # Convert the observation dictionary to tensors
#     obs_tensor = {k: torch.tensor(v, dtype=torch.float32) for k, v in obs.items()}

#     # Perform a few steps in the environment using the policy
#     for _ in range(10):  # Just run for 10 steps for testing
#         # Predict action logits and value from the policy
#         action_logits, value, _ = policy(obs_tensor)

#         # Convert logits to binary actions (thresholding at 0.5 for simplicity)
#         action = (action_logits > 0.5).long().numpy()

#         # Execute the action in the environment
#         obs, reward, done, info = env.step(action)
#         print(f"Action: {action}, Reward: {reward}")

#         # Prepare the next observation
#         obs_tensor = {k: torch.tensor(v, dtype=torch.float32) for k, v in obs.items()}

#         # Optionally update GUI and check for the end of the episode
#         env.render()
#         if done:
#             print("Episode finished. Resetting environment...")
#             obs = env.reset()
#             obs_tensor = {k: torch.tensor(v, dtype=torch.float32) for k, v in obs.items()}