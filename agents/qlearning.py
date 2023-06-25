# agents/qlearning.py

from dataclasses import dataclass, field

import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.experimental.numpy as np

from replay_buffer import ReplayBuffer


@dataclass
class Stats:
    actions: list = field(default_factory=list)
    losses: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    qvalues_0: list = field(default_factory=list)
    qvalues_1: list = field(default_factory=list)

    def clear(self):
        self.losses.clear()
        self.rewards.clear()

    def write_summaries(self, episode_num):
        tf.summary.scalar("avg_action", np.mean(self.actions), step=episode_num)
        tf.summary.scalar("avg_loss", np.mean(self.losses), step=episode_num)
        tf.summary.scalar("sum_reward", np.sum(self.rewards), step=episode_num)
        tf.summary.scalar("avg_qvalue_0", np.mean(self.qvalues_0), step=episode_num)
        tf.summary.scalar("avg_qvalue_1", np.mean(self.qvalues_1), step=episode_num)


@dataclass
class Hyperparameters:
    learning_rate: float = 0.01
    gamma: float = 0.99
    batch_size: int = 256


class QLearningAgent:
    def __init__(self, state_spec, action_spec, replay_buffer_size=2048, params=None):
        state_dim = sum([np.prod(spec.shape) for spec in state_spec.values()])
        self.state_dim = state_dim

        self.action_spec = action_spec
        self.action_dim = action_spec.shape[0]
        self.action_bound = (action_spec.minimum, action_spec.maximum)

        if params is None:
            self.params = Hyperparameters()
        else:
            self.params = params

        # Initialize Q-Network
        # Output is a (action_dim, batch_size) array of Q-values for each action.
        # The value represents the expected return for taking that action in that state.
        self.model = tf.keras.Sequential(
            [
                layers.Dense(
                    16, input_shape=(state_dim,), activation="elu", use_bias=False
                ),
                layers.Dense(8, activation="elu", use_bias=False),
                layers.Dense(2, activation=None, use_bias=False),
            ]
        )

        self.optimizer = tf.keras.optimizers.legacy.Adam(
            learning_rate=float(self.params.learning_rate)
        )

        # Add an attribute to hold loss stats
        self.stats = Stats()

        # Add replay buffer and batch size
        self.replay_buffer = ReplayBuffer(replay_buffer_size, 5)
        # self.batch_size = batch_size

    def get_action(self, state, epsilon=0.0):
        """Given a state, return the agent's action.
        - state: the state the agent is in as a dictionary
        - epsilon: the probability of taking a random action

        The return value should be a numpy array of shape (action_dim,).
        """
        flattened_state = self._flatten_state(state)
        flattened_state = tf.reshape(flattened_state, (1, -1))
        return self.get_action_from_model(
            flattened_state, tf.cast(epsilon, tf.float32)
        ).numpy()

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(1, None), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32),
        ]
    )
    def get_action_from_model(self, state, epsilon):
        """Given a state, return the agent's action.
        - state: the state the agent is in as a tensor
        - epsilon: the probability of taking a random action
        """
        q_values = self.model(state, training=False)

        # Choose a random action with probability epsilon, else choose the action with the highest Q-value
        if tf.random.uniform(()) < epsilon:
            qvalue_argmax = tf.random.uniform(
                shape=(),
                minval=0,
                maxval=2,
                dtype=tf.int32,
            )
        else:
            qvalue_argmax = tf.cast(tf.argmax(q_values[0]), tf.int32)

        action = tf.cond(
            qvalue_argmax == 0,
            lambda: tf.constant([-1], dtype=tf.float32),
            lambda: tf.constant([1], dtype=tf.float32),
        )

        return action

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
        loss, batch_q_values = self.train_step(
            states, actions, next_states, rewards, dones
        )

        # Save the stats
        self.stats.rewards.append(reward.numpy())
        self.stats.losses.append(loss.numpy())
        self.stats.actions.append(action.numpy())
        self.stats.qvalues_0.append(batch_q_values[:, 0].numpy())
        self.stats.qvalues_1.append(batch_q_values[:, 1].numpy())

        return loss

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
        with tf.GradientTape() as tape:
            gamma = tf.cast(self.params.gamma, tf.float32)

            # Calculate Q(s)
            q_values = self.model(state)

            # Calculate Q(s')
            next_q_values = self.model(next_state)

            # Convert action to index
            # The action is a scalar tensor whose values are either -1 or 1.
            # We want to convert it to a 0 or 1 so we can use it to index into the Q-values.
            action_index = tf.cast((action + 1) / 2, tf.int32)
            action_index = tf.squeeze(
                action_index, axis=-1
            )  # remove the extra dimension

            batch_indices = tf.range(tf.shape(action_index)[0])
            gather_indices = tf.stack([batch_indices, action_index], axis=1)

            q_sa = tf.gather_nd(q_values, gather_indices)

            # Calculate max_a' Q(s', a')
            q_s_a_prime = tf.reduce_max(next_q_values, axis=1)

            # Calculate target for Q(s, a)
            q_target_sa = reward + (1.0 - done) * gamma * q_s_a_prime

            # Calculate loss
            loss = tf.reduce_mean(tf.square(q_sa - q_target_sa))

        # Optimize
        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        return loss, q_values

    def write_summaries(self, episode_num):
        self.stats.write_summaries(episode_num)
        self.stats.clear()

    def _flatten_state(self, state):
        return tf.cast(
            tf.concat([state[key] for key in sorted(state.keys())], axis=0), tf.float32
        )
