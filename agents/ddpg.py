# agents/ddpg.py

import tensorflow.experimental.numpy as np

from dataclasses import dataclass, field

import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import HeNormal, Zeros
from tensorflow.keras import layers

from agents.agent import BaseAgent
from agents.ornstein_uhlenbeck_noise import OrnsteinUhlenbeckActionNoise
from replay_buffer import ReplayBuffer


def scale_value(input_value, input_range, target_range):
    """Scale a value from its original range to a target range.

    Args:
        input_value: The input value to be scaled.
        input_range: A tuple specifying the original range of the input.
        target_range: A tuple specifying the target output range.

    Returns:
        The scaled value.
    """
    # Unpack the ranges
    (input_min, input_max) = input_range
    (target_min, target_max) = target_range

    # Scale the input value to the target range
    scaled_value = target_min + (
        (input_value - input_min) / (input_max - input_min)
    ) * (target_max - target_min)

    return scaled_value


@dataclass
class Stats:
    actions: list = field(default_factory=list)
    actor_losses: list = field(default_factory=list)
    critic_losses: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    critic_nmses: list = field(default_factory=list)

    def clear(self):
        self.actions.clear()
        self.actor_losses.clear()
        self.critic_losses.clear()
        self.rewards.clear()
        self.critic_nmses.clear()

    def write_summaries(self, episode_num):
        tf.summary.scalar("avg_action", np.mean(self.actions), step=episode_num)
        tf.summary.scalar(
            "avg_actor_loss", np.mean(self.actor_losses), step=episode_num
        )
        tf.summary.scalar(
            "avg_critic_loss", np.mean(self.critic_losses), step=episode_num
        )
        tf.summary.scalar(
            "avg_critic_nmse", np.mean(self.critic_nmses), step=episode_num
        )
        tf.summary.scalar("sum_reward", np.sum(self.rewards), step=episode_num)


# Actor network or Policy network
# The actor takes a state and returns an action.
# It determines what the next action should be.
def create_actor(state_dim, action_dim, action_bound):
    inputs = layers.Input(shape=(state_dim,))
    x = layers.Dense(
        10,
        kernel_initializer=HeNormal(),
        use_bias=False,
        activation=None,
        # bias_initializer=Zeros(),
        # kernel_regularizer=l2(1e-5),
    )(inputs)
    x = BatchNormalization()(x)
    x = layers.Activation("elu")(x)  # ELU activation here
    x = layers.Dense(
        5,
        kernel_initializer=HeNormal(),
        use_bias=False,
        activation=None,
        # bias_initializer=Zeros(),
        # kernel_regularizer=l2(1e-5),
    )(x)
    x = BatchNormalization()(x)
    x = layers.Activation("elu")(x)  # ELU activation here
    # A value between -1 and 1
    raw_actions = layers.Dense(
        action_dim,
        kernel_initializer=HeNormal(),
        use_bias=False,
        activation="tanh",
    )(x)

    # Scale the output to match the action space
    actions = scale_value(raw_actions, (-1.0, 1.0), action_bound)
    return tf.keras.Model(inputs=inputs, outputs=actions)


# Critic network or Value network or Q-function
# The critic takes a state and an action and returns a q-value.
# The critic observes the action suggested by the actor
# and predicts how good of an action that will be.
def create_critic(state_dim, action_dim):
    state_inputs = layers.Input(shape=(state_dim,))
    action_inputs = layers.Input(shape=(action_dim,))
    inputs = layers.Concatenate()([state_inputs, action_inputs])
    x = layers.Dense(
        10,
        kernel_initializer=HeNormal(),
        use_bias=False,
        activation=None,
        # bias_initializer=Zeros(),
        # kernel_regularizer=l2(1e-5),
    )(inputs)
    x = BatchNormalization()(x)
    x = layers.Activation("elu")(x)  # ELU activation here
    x = layers.Dense(
        5,
        kernel_initializer=HeNormal(),
        use_bias=False,
        activation=None,
        # bias_initializer=Zeros(),
        # kernel_regularizer=l2(1e-5),
    )(x)
    x = BatchNormalization()(x)
    x = layers.Activation("elu")(x)  # ELU activation here
    q_values = layers.Dense(
        1,
        kernel_initializer=HeNormal(),
        use_bias=False,
        activation=None,
    )(x)
    return tf.keras.Model(inputs=[state_inputs, action_inputs], outputs=q_values)


@dataclass
class Hyperparameters:
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 0.001
    critic_lr: float = 0.002
    noise_std: float = 0.1
    batch_size: int = 256


class DDPGAgent(BaseAgent):
    def __init__(self, state_spec, action_spec, replay_buffer_size=65536, params=None):
        state_dim = sum([np.prod(spec.shape) for spec in state_spec.values()])
        self.state_dim = state_dim

        self.action_spec = action_spec
        self.action_dim = action_spec.shape[0]
        self.action_bound = (action_spec.minimum, action_spec.maximum)

        if params is None:
            self.params = Hyperparameters()
        else:
            self.params = params

        # self.memory = TransitionHistory()
        self.replay_buffer = ReplayBuffer(replay_buffer_size, 5)

        # Create networks
        self.actor = create_actor(state_dim, self.action_dim, self.action_bound)
        self.actor_target = create_actor(state_dim, self.action_dim, self.action_bound)
        self.critic = create_critic(state_dim, self.action_dim)
        self.critic_target = create_critic(state_dim, self.action_dim)

        self.stats = Stats()

        # Action noise
        self.noise = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(self.action_dim),
            std_deviation=self.params.noise_std * np.ones(self.action_dim),
        )

        # Optimizers
        initial_actor_lr = self.params.actor_lr  # 0.001
        actor_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_actor_lr,
            decay_steps=100000,
            decay_rate=0.96,
            staircase=True,
        )

        initial_critic_lr = self.params.critic_lr  # 0.002
        critic_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_critic_lr,
            decay_steps=100000,
            decay_rate=0.96,
            staircase=True,
        )

        self.actor_optimizer = tf.keras.optimizers.legacy.Adam(
            learning_rate=actor_lr_schedule
        )
        self.critic_optimizer = tf.keras.optimizers.legacy.Adam(
            learning_rate=critic_lr_schedule
        )

    def get_action(self, state, epsilon=0.0):
        flattened_state = self._flatten_state(state)
        flattened_state = tf.reshape(flattened_state, (1, -1))
        action = self.get_action_from_model(flattened_state, epsilon)
        return action

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(1, None), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32),
        ]
    )
    def get_action_from_model(self, state, epsilon):
        action_min, action_max = self.action_bound

        # Scale the noise to be the size of the action space
        noise = scale_value(self.noise(), (-1.0, 1.0), (action_min, action_max))

        # Get the action from the actor network
        action = self.actor(state)[0] + epsilon * noise

        # Ensure action stays within bounds
        action = tf.clip_by_value(action, action_min, action_max)

        return action

    def get_action_from_model(self, state, epsilon):
        action_min, action_max = self.action_bound

        if tf.random.uniform(()) < epsilon:
            # With probability epsilon, choose a random action
            action = tf.random.uniform(
                shape=(self.action_dim,),
                minval=action_min,
                maxval=action_max,
            )
        else:
            # Otherwise, choose the action from the actor network
            action = self.actor(state)[0]

        return tf.clip_by_value(action, action_min, action_max)

    def train(self, state, action, next_state, reward, done):
        state = self._flatten_state(state)
        next_state = self._flatten_state(next_state)
        reward = tf.constant([reward], dtype=tf.float32)
        done = tf.constant([done], dtype=tf.float32)

        self.replay_buffer.store((state, action, next_state, reward, done))

        # Only start training once enough samples are available in the buffer
        if len(self.replay_buffer) < self.params.batch_size:
            return None

        states, actions, next_states, rewards, dones = self.replay_buffer.sample(
            self.params.batch_size
        )

        training_metrics = self.train_step(
            state=states,
            action=actions,
            next_state=next_states,
            reward=rewards,
            done=dones,
        )

        self.stats.rewards.append(reward.numpy())
        self.stats.actions.append(action.numpy())
        self.stats.actor_losses.append(training_metrics["actor_loss"].numpy())
        self.stats.critic_losses.append(training_metrics["critic_loss"].numpy())
        self.stats.critic_nmses.append(training_metrics["critic_nmse"].numpy())

    @tf.function(
        input_signature=[
            tf.TensorSpec(
                shape=(None, None), dtype=tf.float32
            ),  # state: (state_dim, batch_size)
            tf.TensorSpec(
                shape=(None, None), dtype=tf.float32
            ),  # action: (action_dim, batch_size)
            tf.TensorSpec(
                shape=(None, None), dtype=tf.float32
            ),  # next_state: (state_dim, batch_size)
            tf.TensorSpec(shape=(None, 1), dtype=tf.float32),  # reward
            tf.TensorSpec(shape=(None, 1), dtype=tf.float32),  # done
        ]
    )
    def train_step(self, state, action, next_state, reward, done):
        #
        # Calculate the 'target' q-value using target netowrks
        #

        # Given the next state, pick a next action
        target_actions = self.actor_target(next_state)
        # Then, calculate the q-value for that (next_state, next_action)
        target_q_values = self.critic_target([next_state, target_actions])
        # And add it to the existing reward to get the full reward
        target_q_values = reward + (1.0 - done) * self.params.gamma * target_q_values

        #
        # Update the Critic
        #
        with tf.GradientTape() as tape:
            # Then, we calculate the q-value using the normal critic
            q_values = self.critic([state, action])
            # and we compare it to our target q-value and use that to derive the loss.
            critic_loss = tf.reduce_mean(tf.square(target_q_values - q_values))
            # Calculate this as a metric
            critic_nmse = tf.reduce_mean(
                tf.sqrt(
                    tf.math.divide_no_nan(
                        tf.square(target_q_values - q_values),
                        tf.square(target_q_values),
                    )
                ),
            )
            critic_grads = tape.gradient(critic_loss, self.critic.trainable_weights)

        critic_grads, _ = tf.clip_by_global_norm(critic_grads, clip_norm=1.0)
        self.critic_optimizer.apply_gradients(
            zip(critic_grads, self.critic.trainable_weights)
        )

        #
        # Update the Actor
        #
        with tf.GradientTape() as tape:
            # Get the action for the given state
            actor_actions = self.actor(state)
            # Use the critic to determine the q-value for that action
            q_values = self.critic([state, actor_actions])
            # We define our loss as the negative mean of that q-value
            # to encourage the actor to maximize the q-value
            actor_loss = -tf.reduce_mean(q_values)
            actor_grads = tape.gradient(actor_loss, self.actor.trainable_weights)

        actor_grads, _ = tf.clip_by_global_norm(actor_grads, clip_norm=1.0)
        self.actor_optimizer.apply_gradients(
            zip(actor_grads, self.actor.trainable_weights)
        )

        # Update the target networks
        self._update_target_networks()

        return {
            "critic_loss": critic_loss,
            "critic_nmse": critic_nmse,
            "actor_loss": actor_loss,
        }

    def _update_target_networks(self):
        tau = self.params.tau
        for target, source in zip(self.actor_target.weights, self.actor.weights):
            target.assign(tau * source + (1 - tau) * target)

        for target, source in zip(self.critic_target.weights, self.critic.weights):
            target.assign(tau * source + (1 - tau) * target)

    def write_summaries(self, episode_num):
        self.stats.write_summaries(episode_num)
        self.stats.clear()

    def _flatten_state(self, state):
        return tf.cast(
            tf.concat([state[key] for key in sorted(state.keys())], axis=0), tf.float32
        )
