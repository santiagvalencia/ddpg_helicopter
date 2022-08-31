# Set a random seed for reproducibility
s = 2
# Set pseudo-random generator seeds in Python, NumPy, and TensorFlow
import random
random.seed(s)
import numpy as np
np.random.seed(s)
import tensorflow as tf
tf.random.set_seed(s)

from tensorflow.keras import layers
from typing import Union

class OUActionNoise:
    """
    Ornstein-Uhlenbeck correlated noise. Heavily based on Hemant Singh's implementation of DDPG
    in Keras: https://keras.io/examples/rl/ddpg_pendulum/
    """
    def __init__(self, mean: float, std_deviation: float, theta: float=0.15, 
                dt: float=1e-2, x_initial: Union[np.ndarray, list, None]=None):
        # Set noise amplitude
        self.theta = theta
        # Set noise mean
        self.mean = mean
        # Set noise standard deviation
        self.std_dev = std_deviation
        # Set noise time step
        self.dt = dt
        # Set initial state
        self.x_initial = x_initial
        # Reset the noise object
        self.reset()

    def __call__(self) -> Union[np.ndarray, list]:
        """
        Determine the noise signal when calling an instance of this class.
        Based on Hemant Singh's implementation of DDPG in Keras: 
        https://keras.io/examples/rl/ddpg_pendulum/ and on the discretization of
        the Ornstein-Uhlenbeck process described in:
        https://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
        Returns:
            Union[np.ndarray, list]: Correlated noise signal
        """
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape, scale=self.std_dev)
        )

        # Store previous noise signal for later use
        self.x_prev = x

        return x

    def reset(self):
        """
        Reset the noise object
        """
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)

class Buffer:
    """
    Replay buffer from which to draw experiences and learn both the critic function 
    and the actor policy.
    Heavily based on Hemant Singh's implementation of DDPG
    in Keras: https://keras.io/examples/rl/ddpg_pendulum/

    """
    def __init__(self, num_states: int, num_actions: int, 
                buffer_capacity: int=100000, batch_size: int=64):
        """
        Initialize instance of the Buffer class

        Args:
            num_states (int): Dimension of state space
            num_actions (int): Dimension of action space
            buffer_capacity (int, optional): Maximum number of transitions to store. Defaults to 100000.
            batch_size (int, optional): Size of the training mini-batches. Defaults to 64.
        """
        # Maximum number of transitions to store
        self.buffer_capacity = buffer_capacity
        # Size of the training mini-batch
        self.batch_size = batch_size

        # Count the number of records (can exceed self.buffer_capacity)
        self.buffer_counter = 0

        # Create one buffer per element in the (s, a, r, s') transition
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    def record(self, obs_tuple: tuple):
        """
        Save a (s, a, r, s') transition to the buffer

        Args:
            obs_tuple (tuple): (s, a, r, s') transition
        """

        # Create an index to replace either the oldest record or 
        # a random record with 0.8 and 0.2 probability, respectively
        if self.buffer_counter == 0:
            index = 0
        else:
            index = np.random.choice([self.buffer_counter % self.buffer_capacity, np.random.randint(0, min(self.buffer_counter, self.buffer_capacity))], p=[0.8, 0.2])

        # Store the components of the tuple in their respective buffers
        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        # Add to the buffer counter
        self.buffer_counter += 1

    @tf.function
    def update(
        self, gamma: float, target_actor: tf.keras.Model, target_critic: tf.keras.Model,
        critic_model: tf.keras.Model, critic_optimizer: tf.keras.optimizers.Optimizer,
        actor_model: tf.keras.Model, actor_optimizer: tf.keras.optimizers.Optimizer, 
        state_batch: tf.Tensor, action_batch: tf.Tensor, reward_batch: tf.Tensor, next_state_batch: tf.Tensor
    ):
        """
        Update actor and critic networks' weights according to the DDPG algorithm
        Args:
            gamma (float): Discount rate for future rewards
            target_actor (tf.keras.Model): Target actor network
            target_critic (tf.keras.Model): Target critic network
            critic_model (tf.keras.Model): Main critic network
            critic_optimizer (tf.keras.optimizers.Optimizer): Optimizer for the critic network
            actor_model (tf.keras.Model): Main actor network
            actor_optimizer (tf.keras.optimizers.Optimizer): Optimizer for the actor network
            state_batch (tf.Tensor): Mini-batch of states
            action_batch (tf.Tensor): Mini-batch of actions
            reward_batch (tf.Tensor): Mini-batch of rewards
            next_state_batch (tf.Tensor): Mini-batch of next states
        """
        # Track operations for automatic differentiation
        with tf.GradientTape() as tape:
            # Find the actions taken by the target actor
            target_actions = target_actor(next_state_batch, training=True)
            # Calculate the y term from the DDPG algorithm
            y = reward_batch + gamma * target_critic(
                [next_state_batch, target_actions], training=True
            )
            # Calculate the main critic's values assigned to the target's chosen actions
            critic_value = critic_model([state_batch, action_batch], training=True)
            # Calculate the critic loss from the DDPG algorithm
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        # Take the gradient of the critic loss
        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        # Clip the gradient to avoid exploding gradients
        critic_grad, _ = tf.clip_by_global_norm(critic_grad, 5.0)
        # Apply the gradients to the critic optimizer
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            # Get the actions chosen by the actor given the mini-batch of states
            actions = actor_model(state_batch, training=True)
            # Get the critic's value of the states and actor's chosen actions
            critic_value = critic_model([state_batch, actions], training=True)
            # The aim when training the actor is to maximize the critic's value, 
            # which is equivalent to minimizing its negative
            actor_loss = -tf.math.reduce_mean(critic_value)

        # Take the gradient of the actor loss
        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        # Clip the gradient to avoid exploding gradients
        actor_grad, _ = tf.clip_by_global_norm(actor_grad, 5.0)
        # Apply the gradients to the actor optimizer
        actor_optimizer.apply_gradients(
            zip(actor_grad, actor_model.trainable_variables)
        )

    def learn(self, gamma: float, target_actor: tf.keras.Model, 
             target_critic: tf.keras.Model, critic_model: tf.keras.Model,
             critic_optimizer: tf.keras.optimizers.Optimizer,
             actor_model: tf.keras.Model, 
             actor_optimizer: tf.keras.optimizers.Optimizer):
        """
        Carry out offline learning from the replay buffer

        Args:
            gamma (float): Discount rate for future rewards
            target_actor (tf.keras.Model): Target actor model
            target_critic (tf.keras.Model): Target critic model
            critic_model (tf.keras.Model): Main critic model
            critic_optimizer (tf.keras.optimizers.Optimizer): Optimizer for the critic model
            actor_model (tf.keras.Model): Main actor model
            actor_optimizer (tf.keras.optimizers.Optimizer): Optimizer for the actor model
        """
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Draw a random set of indices to form the minibatches
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert minibatches to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        # Update main critic and actor networks' weights
        self.update(gamma, target_actor, target_critic,
        critic_model, critic_optimizer,
        actor_model, actor_optimizer, state_batch, action_batch, reward_batch, next_state_batch)


@tf.function
def update_target(target_weights: list, weights: list, tau: float):
    """
    Update the target networks with an update rate tau.
    Taken from Hemant Singh's implementation of DDPG
    in Keras: https://keras.io/examples/rl/ddpg_pendulum/

    Args:
        target_weights (list): List of target network's weights
        weights (list): List of main network's weights
        tau (float): Update rate (<< 1)
    """
    for (a, b) in zip(target_weights, weights):
        # Update the weights 
        a.assign(b*tau + a*(1 - tau))

def get_actor(num_states: int, num_actions: int, upper_bound: float) -> tf.keras.Model:
    """
    Create an actor neural network. Heavily based on Hemant Singh's implementation of DDPG
    in Keras: https://keras.io/examples/rl/ddpg_pendulum/

    Args:
        num_states (int): Dimension of the state space
        num_actions (int): Dimension of the action space
        upper_bound (float): Upper bound of the actions (actions are assumed symmetrical around 0)

    Returns:
        tf.keras.Model: Actor neural network
    """
    # Initialize weights between -3e-3 and 3-e3 to make sure gradients do not go to zero in first steps
    last_init = tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)

    # Take the state as input
    inputs = layers.Input(shape=(num_states,), name='State')
    # Add two dense, ReLU-activated layers 
    out = layers.Dense(256, activation="relu", name='Hidden_1')(inputs)
    out = layers.Dense(256, activation="relu", name='Hidden_2')(out)
    # Add a layer that gives the action (normalized between -1 and 1)
    outputs = layers.Dense(num_actions, activation="tanh", kernel_initializer=last_init, name='Normalized_action')(out)

    # De-normalize the action with the upper bound of the action space
    outputs = outputs * upper_bound
    # Create and return the full model
    model = tf.keras.Model(inputs, outputs)
    return model


def get_critic(num_states: int, num_actions: int) -> tf.keras.Model:
    """
    Create a critic neural network. Heavily based on Hemant Singh's implementation of DDPG
    in Keras: https://keras.io/examples/rl/ddpg_pendulum/

    Args:
        num_states (int): Dimension of the state space
        num_actions (int): Dimension of the action space

    Returns:
        tf.keras.Model: Critic neural network
    """
    # Take the state as an input
    state_input = layers.Input(shape=(num_states), name='State')
    # Add three dense, ReLU-activated layers
    state_out = layers.Dense(16, activation="relu", name='Hidden_1')(state_input)
    state_out = layers.Dense(16, activation="relu", name='Hidden_2')(state_out)
    state_out = layers.Dense(32, activation="relu", name='Hidden_3')(state_out)

    # Take the action as an input
    action_input = layers.Input(shape=(num_actions), name='Action')
    # Add two dense, ReLU-activated layers
    action_out = layers.Dense(32, activation="relu", name='Hidden_4')(action_input)
    action_out = layers.Dense(32, activation="relu", name='Hidden_5')(action_out)

    # Concatenate the state and action 'branches'
    concat = layers.Concatenate(name='Concatenate')([state_out, action_out])

    # Add two fully connected, ReLU-activated layers
    out = layers.Dense(256, activation="relu", name='Hidden_6')(concat)
    out = layers.Dense(256, activation="relu", name='Hidden_7')(out)
    # Give the Q value of the state, action pair as output
    outputs = layers.Dense(1, name='Q_value')(out)

    # Create and return the full model
    model = tf.keras.Model([state_input, action_input], outputs)

    return model

def policy(state: tf.Tensor, actor_model: tf.keras.Model,
         noise_object: OUActionNoise, add_noise: bool=True, 
         lower_bound: float=-np.pi/2, upper_bound: float=np.pi/2) -> np.ndarray:
    """
    Policy function that add optionally adds noise to the actor model's chosen action
    Heavily based on Hemant Singh's implementation of DDPG
    in Keras: https://keras.io/examples/rl/ddpg_pendulum/

    Args:
        state (tf.Tensor): Current state (has to be a tf.Tensor)
        actor_model (tf.keras.Model): Actor model that maps actions to states
        noise_object (OUActionNoise): Correlated noise object
        add_noise (bool, optional): Whether to add noise to the policy. Defaults to True.
        lower_bound (float, optional): Lower bound for clipping. Defaults to -np.pi/2.
        upper_bound (float, optional): Upper bound for clipping. Defaults to np.pi/2.

    Returns:
        np.ndarray: Chosen action given the input state
    """
    # Get action out of the current state
    sampled_actions = tf.squeeze(actor_model(state))
    # Call the noise object to get the noise signal
    noise = noise_object()
    # Optionally add the noise to the selected action
    if add_noise:
        sampled_actions = sampled_actions.numpy() + noise
    else:
        sampled_actions = sampled_actions.numpy()

    # Clip the action to make it fit within the specified bounds
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)

    return np.squeeze(legal_action)