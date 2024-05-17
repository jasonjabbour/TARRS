import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.environments import gym_wrapper
from tf_agents.agents.ppo import ppo_agent
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import value_network
from tf_agents.utils import common
from tf_agents.trajectories import trajectory
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.networks import network
from tf_agents.networks import categorical_projection_network
from tf_agents.specs import BoundedTensorSpec

from spanning_tree_env import SpanningTreeEnv

# This function wraps the custom Gym environment to make it compatible with TF-Agents
def create_tf_env(environment):
    # Convert a Gym environment into a TFPyEnvironment which is TensorFlow friendly.
    return tf_py_environment.TFPyEnvironment(gym_wrapper.GymWrapper(environment))

class CustomCategoricalProjectionNetwork(network.Network):
    '''
        Handles different ranges od discrete actions for each dimension by
        using separate CategoricalProjectNetwork instances for each action
        dimension. Outputs from these networks are concatenated to match
        the expected action space of the agent. 
    '''
    def __init__(self, action_spec, name='CustomCategoricalProjectionNetwork'):
        '''
            The network first calculates the number of actions for each
            dimension using the minimum and maximum values from the 
            action_spec. It then creates a separate CategoricalProjectionNetwork 
            for each dimension based on these numbers.
        '''
        super(CustomCategoricalProjectionNetwork, self).__init__(
            input_tensor_spec=tf.TensorSpec(shape=[None], dtype=tf.float32),
            state_spec=(), name=name)

        # Calculate the number of actions for each dimension
        self._num_actions_per_dim = action_spec.maximum - action_spec.minimum + 1

        # Create a CategoricalProjectionNetwork for each action dimension
        self._projection_networks = [
            categorical_projection_network.CategoricalProjectionNetwork(
                sample_spec=BoundedTensorSpec(
                    shape=(), 
                    dtype=tf.int32, 
                    minimum=0, 
                    maximum=num_actions-1
                )
            ) for num_actions in self._num_actions_per_dim
        ]
        
    def call(self, inputs, step_type=None, network_state=(), training=False):
        '''
            During the forward pass (call method), it processes inputs 
            through each projection network and concatenates their 
            outputs. Each projection network handles one dimension of 
            the action space, ensuring that actions are sampled correctly 
            according to their respective ranges.
        '''
        outputs = [proj_net(inputs)[0] for proj_net in self._projection_networks]
        concatenated_outputs = tf.concat(outputs, axis=-1)
        return concatenated_outputs, network_state

# This function creates a PPO Agent
def create_agent(train_env):

    # Define preprocessing combiner to combine different parts of observation.
    preprocessing_combiner = tf.keras.layers.Concatenate(axis=-1)

    # Initialize the ActorDistributionNetwork with the environment specs and the custom projection network.
    actor_net = actor_distribution_network.ActorDistributionNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        fc_layer_params=(100, 50),  # Specify the sizes of the fully connected layers.
        projection_network=CustomCategoricalProjectionNetwork(train_env.action_spec())  # Use the custom projection network.
    )

    # Actor network, which outputs actions given environmental observations
    actor_net = actor_distribution_network.ActorDistributionNetwork(
        train_env.observation_spec(),  # Describes the structure of the observation space from the environment
        train_env.action_spec(),       # Describes the structure of the action space
        preprocessing_combiner=preprocessing_combiner,
        fc_layer_params=(100, 50),      # The number and size of hidden layers in the neural network
        projection_network=CustomCategoricalProjectionNetwork(train_env.action_spec())  # Use the custom projection network.
        )     

    # Value network, which estimates the value (expected total future reward) of each observation
    value_net = value_network.ValueNetwork(
        train_env.observation_spec(),
        preprocessing_combiner=preprocessing_combiner,
        fc_layer_params=(100, 50))

    # Optimizer to use for training the agent, Adam is a common choice
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)

    # Counter to keep track of training steps
    train_step_counter = tf.Variable(0)

    # Create the PPO Agent
    agent = ppo_agent.PPOAgent(
        train_env.time_step_spec(),  # Describes the time step structure of the environment
        train_env.action_spec(),
        optimizer,
        actor_net=actor_net,
        value_net=value_net,
        num_epochs=10,               # Number of times the collected data is used to update the network
        train_step_counter=train_step_counter)
    agent.initialize()

    return agent

# This function defines the training process
def train_agent(num_iterations, collect_episodes_per_iteration):
    # Create an instance of the environment
    env = SpanningTreeEnv(min_nodes=5, max_nodes=15, min_redundancy=2, max_redundancy=4)
    tf_env = create_tf_env(env)
    agent = create_agent(tf_env)

    # Create a buffer that stores trajectories collected during training
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,  # Specifies the structure of data the buffer will hold
        batch_size=tf_env.batch_size,
        max_length=1000)  # Maximum number of items (trajectories) the buffer can hold

    # Reset the environment and get the initial time_step
    time_step = tf_env.reset()
    policy_state = agent.collect_policy.get_initial_state(tf_env.batch_size)
    
    # Run the training for a specified number of iterations
    for _ in range(num_iterations):
        # Collect data for each episode
        for _ in range(collect_episodes_per_iteration):
            # The agent decides an action based on the policy
            action_step = agent.collect_policy.action(time_step, policy_state)  
            # The environment responds to the action
            next_time_step = tf_env.step(action_step.action)  
            # Create a trajectory from the transitions
            traj = trajectory.from_transition(time_step, action_step, next_time_step)  
            # Add the trajectory to the replay buffer
            replay_buffer.add_batch(traj)  
            time_step = next_time_step
            policy_state = action_step.state
        
        # Prepare the data for training
        dataset = replay_buffer.as_dataset(
            num_parallel_calls=3, 
            sample_batch_size=64, 
            num_steps=2).prefetch(3)  # Prepare batches of data for training

        iterator = iter(dataset)
        
        # Training step
        for _ in range(agent.train_steps_per_iteration):
            # Get a batch of experiences from the buffer
            experience, _ = next(iterator)  
            # Train the agent using the experiences
            train_loss = agent.train(experience).loss  

        print('Iteration = {0}: Loss = {1}'.format(_, train_loss)) 


if __name__ == "__main__":
    # Number of iterations for training
    num_iterations = 10000  
    # Number of episodes to collect data from before updating the model
    collect_episodes_per_iteration = 5  

    train_agent(num_iterations, collect_episodes_per_iteration)    