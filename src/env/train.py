import os
import gymnasium as gym
import torch
import time
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from spanning_tree_env import SpanningTreeEnv

MIN_NODES = 30
MAX_NODES = 30
MIN_REDUNDANCY = 3
MAX_REDUNDANCY = 4
NUM_ATTACKED_NODES = 1
TRAINING_MODE = False
RENDER_EVAL_ENV = True
TOTAL_TIMESTEPS = 10000000
MODEL_DIR_BASE = "./models"
MODEL_PATH_4_INFERENCE = "./models/model5/ppo_spanning_tree_final"

def create_incremental_dir(base_path, prefix="model"):
    """Create a directory with an incrementing index to avoid overwriting previous models."""
    index = 1
    while True:
        model_dir = os.path.join(base_path, f"{prefix}{index}")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            return model_dir
        index += 1

def train(env, eval_env, total_timesteps, model_dir_base):
    """Train the model."""
    model_dir = create_incremental_dir(model_dir_base)  # Create an incrementally named directory for this training run
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")
    model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log="./tensorboard_logs/", device=device)

    # Setup checkpoint every set number of steps
    checkpoint_callback = CheckpointCallback(save_freq=100000, save_path=os.path.join(model_dir, 'checkpoints/'), name_prefix='ppo_spanning_tree')

    # Setup Eval Callback
    eval_callback = EvalCallback(eval_env, best_model_save_path=os.path.join(model_dir, 'best_model/'),
                                 log_path=os.path.join(model_dir, 'logs/'), eval_freq=100000,
                                 deterministic=True, render=False)

    callback = CallbackList([checkpoint_callback, eval_callback])

    # Training the model with callbacks
    model.learn(total_timesteps=total_timesteps, callback=callback)
    model.save(os.path.join(model_dir, "ppo_spanning_tree_final"))  # Saving final model state after training
    return model

def test(env, model_path):
    """Test the model with visualization."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing on device: {device}")
    model = PPO.load(model_path, env=env, device=device)
    obs = env.reset()
    total_reward = 0
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        print(action, reward)
        total_reward += reward
        if done:
            break  # Exit the loop when the episode is done
    print(f"Total Reward: {total_reward}")

def main():
    print("CUDA available:", torch.cuda.is_available())
    render_mode = True if not TRAINING_MODE else False
    n_envs = 1 if not TRAINING_MODE else 20

    env = make_vec_env(lambda: SpanningTreeEnv(min_nodes=MIN_NODES, 
                                               max_nodes=MAX_NODES, 
                                               min_redundancy=MIN_REDUNDANCY, 
                                               max_redundancy=MAX_REDUNDANCY, 
                                               num_attacked_nodes=NUM_ATTACKED_NODES, 
                                               render_mode=render_mode), 
                                               n_envs=n_envs)

    if TRAINING_MODE:

        eval_env = make_vec_env(lambda: SpanningTreeEnv(min_nodes=MIN_NODES, 
                                                max_nodes=MAX_NODES, 
                                                min_redundancy=MIN_REDUNDANCY, 
                                                max_redundancy=MAX_REDUNDANCY, 
                                                num_attacked_nodes=NUM_ATTACKED_NODES, 
                                                render_mode=RENDER_EVAL_ENV), 
                                                n_envs=5)
                                                
        train(env, eval_env, TOTAL_TIMESTEPS, MODEL_DIR_BASE)
    else:
        test(env, MODEL_PATH_4_INFERENCE)  # Specify the correct path for the tested model

if __name__ == '__main__':
    main()









# import tensorflow as tf
# from tf_agents.environments import tf_py_environment
# from tf_agents.environments import gym_wrapper
# from tf_agents.agents.ppo import ppo_agent
# from tf_agents.networks import actor_distribution_network
# from tf_agents.networks import value_network
# from tf_agents.utils import common
# from tf_agents.trajectories import trajectory
# from tf_agents.replay_buffers import tf_uniform_replay_buffer
# from tf_agents.networks import network
# from tf_agents.networks import categorical_projection_network
# from tf_agents.specs import BoundedTensorSpec

# from spanning_tree_env import SpanningTreeEnv

# # This function wraps the custom Gym environment to make it compatible with TF-Agents
# def create_tf_env(environment):
#     # Convert a Gym environment into a TFPyEnvironment which is TensorFlow friendly.
#     return tf_py_environment.TFPyEnvironment(gym_wrapper.GymWrapper(environment))

# class CustomCategoricalProjectionNetwork(network.Network):
#     '''
#     Handles different ranges of discrete actions for each dimension by
#     using separate CategoricalProjectionNetwork instances for each action
#     dimension. Outputs from these networks are concatenated to match
#     the expected action space of the agent.
#     '''
#     def __init__(self, action_spec, name='CustomCategoricalProjectionNetwork'):
#         super(CustomCategoricalProjectionNetwork, self).__init__(
#             input_tensor_spec=tf.TensorSpec(shape=[None], dtype=tf.float32),
#             state_spec=(), name=name)

#         # Assume action_spec is an instance of MultiDiscrete space from Gym, manually defined
#         # For a MultiDiscrete space defined as [2, num_nodes, num_nodes], we manually set up:
#         num_dimensions = [2, action_spec.maximum[1] + 1, action_spec.maximum[2] + 1]

#         self._projection_networks = [
#             categorical_projection_network.CategoricalProjectionNetwork(
#                 sample_spec=BoundedTensorSpec(
#                     shape=(),
#                     dtype=tf.int32,
#                     minimum=0,
#                     maximum=num_actions - 1
#                 )
#             ) for num_actions in num_dimensions
#         ]

#     def call(self, inputs, step_type=None, network_state=(), training=False):
#         # Flatten inputs if they come batched
#         inputs = tf.cast(inputs, dtype=tf.float32)
#         outputs = [proj_net(inputs[:, i:i+1])[0] for i, proj_net in enumerate(self._projection_networks)]
#         concatenated_outputs = tf.concat(outputs, axis=-1)
#         return concatenated_outputs, network_state

# def create_agent(train_env):
#     '''
#     This function creates a PPO Agent with a custom action projection network.
#     The actor and value networks are configured with specific layer parameters
#     and a custom preprocessing combiner that handles the agent's observations.
#     '''
#     preprocessing_combiner = tf.keras.layers.Concatenate(axis=-1)

#     # Custom network for action projection
#     custom_proj_net = CustomCategoricalProjectionNetwork(train_env.action_spec())

#     # Actor network, which outputs actions given environmental observations
#     actor_net = actor_distribution_network.ActorDistributionNetwork(
#         train_env.observation_spec(),
#         train_env.action_spec(),
#         preprocessing_combiner=preprocessing_combiner,
#         fc_layer_params=(100, 50),  # Specify the sizes of the fully connected layers.
#         discrete_projection_net=custom_proj_net,
#     )

#     # Value network, which estimates the value (expected total future reward) of each observation
#     value_net = value_network.ValueNetwork(
#         train_env.observation_spec(),
#         preprocessing_combiner=preprocessing_combiner,
#         fc_layer_params=(100, 50))

#     # Optimizer to use for training the agent, Adam is a common choice
#     optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

#     # Counter to keep track of training steps
#     train_step_counter = tf.Variable(0)

#     # Create the PPO Agent
#     agent = ppo_agent.PPOAgent(
#         train_env.time_step_spec(),  # Describes the time step structure of the environment
#         train_env.action_spec(),
#         optimizer,
#         actor_net=actor_net,
#         value_net=value_net,
#         num_epochs=10,               # Number of times the collected data is used to update the network
#         train_step_counter=train_step_counter)
#     agent.initialize()

#     return agent

# # This function defines the training process
# def train_agent(num_iterations, collect_episodes_per_iteration):
#     # Create an instance of the environment
#     env = SpanningTreeEnv(min_nodes=5, max_nodes=15, min_redundancy=2, max_redundancy=4)
#     tf_env = create_tf_env(env)
#     agent = create_agent(tf_env)

#     # Create a buffer that stores trajectories collected during training
#     replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
#         data_spec=agent.collect_data_spec,  # Specifies the structure of data the buffer will hold
#         batch_size=tf_env.batch_size,
#         max_length=1000)  # Maximum number of items (trajectories) the buffer can hold

#     # Reset the environment and get the initial time_step
#     time_step = tf_env.reset()
#     policy_state = agent.collect_policy.get_initial_state(tf_env.batch_size)
    
#     # Run the training for a specified number of iterations
#     for _ in range(num_iterations):
#         # Collect data for each episode
#         for _ in range(collect_episodes_per_iteration):
#             # The agent decides an action based on the policy
#             action_step = agent.collect_policy.action(time_step, policy_state)  
#             # The environment responds to the action
#             next_time_step = tf_env.step(action_step.action)  
#             # Create a trajectory from the transitions
#             traj = trajectory.from_transition(time_step, action_step, next_time_step)  
#             # Add the trajectory to the replay buffer
#             replay_buffer.add_batch(traj)  
#             time_step = next_time_step
#             policy_state = action_step.state
        
#         # Prepare the data for training
#         dataset = replay_buffer.as_dataset(
#             num_parallel_calls=3, 
#             sample_batch_size=64, 
#             num_steps=2).prefetch(3)  # Prepare batches of data for training

#         iterator = iter(dataset)
        
#         # Training step
#         for _ in range(agent.train_steps_per_iteration):
#             # Get a batch of experiences from the buffer
#             experience, _ = next(iterator)  
#             # Train the agent using the experiences
#             train_loss = agent.train(experience).loss  

#         print('Iteration = {0}: Loss = {1}'.format(_, train_loss)) 


# if __name__ == "__main__":
#     # Number of iterations for training
#     num_iterations = 10000  
#     # Number of episodes to collect data from before updating the model
#     collect_episodes_per_iteration = 5  

#     train_agent(num_iterations, collect_episodes_per_iteration)    