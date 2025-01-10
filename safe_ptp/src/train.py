import os
import gymnasium as gym
import torch
import time

from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Logger

from safe_ptp.src.env.spanning_tree_env import SpanningTreeEnv
from safe_ptp.src.alg.custom_gcn_policy import CustomGNNActorCriticPolicy
from safe_ptp.src.alg.custom_gcn_policy_static import CustomGNNActorCriticPolicyStatic


START_DIFFICULTY_LEVEL = 16
FINAL_DIFFICULTY_LEVEL = 16
MIN_NODES = 20
MAX_NODES = 20
MIN_REDUNDANCY = 3
TRAINING_MODE = False
RENDER_EVAL_ENV = True
SHOW_WEIGHT_LABELS = False
TOTAL_TIMESTEPS = 30000000
MODEL_DIR_BASE = "./models"
ALGO = 'PPO'
# MODEL_PATH_4_INFERENCE = "./models/model14/best_model/best_model"
MODEL_PATH_4_INFERENCE = f"./models/model14/checkpoints/{ALGO.lower()}_spanning_tree_30000000_steps"

class DifficultyLevelLoggingCallback(BaseCallback):
    def __init__(self, eval_freq, verbose=0):
        super(DifficultyLevelLoggingCallback, self).__init__(verbose)
        self.eval_freq = eval_freq  
    
    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq == 0:
            # Get attributes for env 0 using get_attr with correct indices
            current_level_env0 = self.training_env.get_attr('current_level', indices=[0])[0]
            current_level_timesteps_env0 = self.training_env.get_attr('current_level_total_timesteps', indices=[0])[0]
            performance_env0 = self.training_env.get_attr('get_level_average_performance', indices=[0])[0]

            # Get cumulative data across all environments
            current_level_list = self.training_env.get_attr('current_level')
            level_timesteps_list = self.training_env.get_attr('current_level_total_timesteps')
            level_performance_list = self.training_env.get_attr('get_level_average_performance')

            # Calculate averages and cumulative metrics
            avg_current_level = sum(current_level_list) / len(current_level_list) if current_level_list else 0
            cumulative_timesteps = sum(level_timesteps_list)
            average_performance = sum(level_performance_list) / len(level_performance_list) if level_performance_list else 0

            # Use the built-in logger to output to Tensorboard
            logger = self.logger if self.logger is not None else Logger.DEFAULT
            logger.record("Curriculum/average_current_level", avg_current_level)
            logger.record("Curriculum/cumulative_total_timesteps", cumulative_timesteps)
            logger.record("Curriculum/avg_performance", average_performance)
            logger.record("Curriculum/env_0_current_level", current_level_env0)
            logger.record("Curriculum/env_0_current_level_timesteps", current_level_timesteps_env0)
            logger.record("Curriculum/env_0_avg_performance", performance_env0)
            logger.dump(self.num_timesteps)
        
        return True

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

    if ALGO == 'PPO':
        # MLP architecture
        policy_kwargs = dict(
            net_arch=[512, 512, 256, 128]  
        )

        model = PPO("MlpPolicy",
                    env, 
                    verbose=1, 
                    policy_kwargs=policy_kwargs,
                    tensorboard_log="./tensorboard_logs/", 
                    device=device, 
                    learning_rate=0.0003, 
                    clip_range=0.1,
                    batch_size=64)
    elif ALGO == 'SAC':
        model = SAC(CustomGNNActorCriticPolicyStatic, 
                    env, 
                    verbose=1, 
                    tensorboard_log="./tensorboard_logs/", 
                    device=device,
                    learning_rate=0.0003, 
                    batch_size=64, 
                    buffer_size=100000, 
                    ent_coef='auto', 
                    train_freq=(32, 'step'),  # Train after 64 steps
                    gradient_steps=64,  # Number of gradient steps after collecting new samples
                    use_sde=True, 
                    sde_sample_freq=4, 
                    target_update_interval=1)

    # Setup checkpoint every set number of steps
    checkpoint_callback = CheckpointCallback(save_freq=100000, save_path=os.path.join(model_dir, 'checkpoints/'), name_prefix='ppo_spanning_tree')

    # Setup Eval Callback
    eval_callback = EvalCallback(eval_env, best_model_save_path=os.path.join(model_dir, 'best_model/'),
                                 log_path=os.path.join(model_dir, 'logs/'), eval_freq=100000,
                                 deterministic=True, render=False)

    # Callback for Logging Difficulty Level
    difficulty_logging_callback = DifficultyLevelLoggingCallback(eval_freq=100000)

    callback = CallbackList([checkpoint_callback, eval_callback, difficulty_logging_callback])

    # Training the model with callbacks
    model.learn(total_timesteps=total_timesteps, callback=callback)
    model.save(os.path.join(model_dir, f"{ALGO.lower()}_spanning_tree_final"))  # Saving final model state after training
    return model

def test(env, model_path):
    """Test the model with visualization."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing on device: {device}")

    if ALGO == 'PPO':
        model = PPO.load(model_path, env=env, device=device)
    elif ALGO == 'SAC':
        model = SAC.load(model_path, env=env, device=device)

    obs = env.reset()
    obs = env.reset()
    # obs = env.reset()
    # obs = env.reset()
    # obs = env.reset()
    print("Attacked Network...")
    print("Timing:")
    # print(env.envs[0].calculate_reward())
    time.sleep(10)
    total_reward = 0
    while True:
        action, _states = model.predict(obs, deterministic=False)
        obs, reward, done, info = env.step(action)
        print(action, reward)
        total_reward += reward
        print("Pausing...")
        time.sleep(5)
        if done:
            time.sleep(30)
            break  # Exit the loop when the episode is done
    print(f"Total Reward: {total_reward}")

def main():
    print("CUDA available:", torch.cuda.is_available())
    render_mode = True if not TRAINING_MODE else False
    n_envs = 1 if not TRAINING_MODE else 1

    env = make_vec_env(lambda: SpanningTreeEnv(min_nodes=MIN_NODES, 
                                               max_nodes=MAX_NODES, 
                                               min_redundancy=MIN_REDUNDANCY, 
                                               start_difficulty_level=START_DIFFICULTY_LEVEL, 
                                               final_difficulty_level=FINAL_DIFFICULTY_LEVEL,
                                               render_mode=RENDER_EVAL_ENV, 
                                               show_weight_labels=SHOW_WEIGHT_LABELS), 
                                               n_envs=n_envs)

    if TRAINING_MODE:

        # TODO FIX TO CHANGE DIFFICULTY LEVELS
        eval_env = make_vec_env(lambda: SpanningTreeEnv(min_nodes=MIN_NODES, 
                                                max_nodes=MAX_NODES, 
                                                min_redundancy=MIN_REDUNDANCY, 
                                                start_difficulty_level=START_DIFFICULTY_LEVEL, 
                                                final_difficulty_level=FINAL_DIFFICULTY_LEVEL,
                                                render_mode=RENDER_EVAL_ENV), 
                                                n_envs=1)
                                                
        train(env, eval_env, TOTAL_TIMESTEPS, MODEL_DIR_BASE)
    else:
        test(env, MODEL_PATH_4_INFERENCE)  # Specify the correct path for the tested model

if __name__ == '__main__':
    main()


