import tensorflow as tf
from tf_agents.environments import suite_gym
from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.utils import common
from tf_agents.environments import tf_py_environment

def main():
    # Set up the CartPole environment from Gym suite
    env = suite_gym.load('CartPole-v0')
    tf_env = tf_py_environment.TFPyEnvironment(env)

    # Create a QNetwork for the DQN agent
    q_net = q_network.QNetwork(
        tf_env.observation_spec(),
        tf_env.action_spec(),
        fc_layer_params=(100,)
    )

    # Set up the DQN agent
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001)
    train_step_counter = tf.Variable(0)
    agent = dqn_agent.DqnAgent(
        tf_env.time_step_spec(),
        tf_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=train_step_counter
    )
    agent.initialize()

    # Reset the environment and get the initial time_step
    time_step = tf_env.reset()

    # Convert the time_step observation to a batched tensor
    # This is where we ensure the observation is batched
    observation_batch = tf.expand_dims(time_step.observation, axis=0)

    # Use the agent's policy to compute the action
    action_step = agent.policy.action(time_step)

    # Execute one step in the environment
    next_time_step = tf_env.step(action_step.action)

    print("Initial test completed successfully!")

if __name__ == "__main__":
    main()
