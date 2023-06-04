# agents/qlearning.py

from dataclasses import dataclass, field

import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.experimental.numpy as np
from agents.ornstein_uhlenbeck_noise import OrnsteinUhlenbeckActionNoise


@dataclass
class Stats:
    actions: list = field(default_factory=list)
    losses: list = field(default_factory=list)
    episode_rewards: list = field(default_factory=list)
    qvalues_0: list = field(default_factory=list)
    qvalues_1: list = field(default_factory=list)

    def clear(self):
        self.losses.clear()
        self.episode_rewards.clear()

    def write_summaries(self, episode_num):
        tf.summary.scalar("avg_action", np.mean(self.actions), step=episode_num)
        tf.summary.scalar("avg_loss", np.mean(self.losses), step=episode_num)
        tf.summary.scalar(
            "episode_reward", np.sum(self.episode_rewards), step=episode_num
        )
        tf.summary.scalar("avg_qvalue_0", np.mean(self.qvalues_0), step=episode_num)
        tf.summary.scalar("avg_qvalue_1", np.mean(self.qvalues_1), step=episode_num)


@dataclass
class Hyperparameters:
    learning_rate: float = 0.01
    gamma: float = 0.99
    # epsilon: float = 0.1
    # noise_std: float = 0.1
    # batch_size: int = 256
    # memory_capacity: int = int(1e6)


class QLearningAgent:
    def __init__(self, state_dim, action_spec, params=None):
        self.state_dim = state_dim

        # self.env.action_spec()

        self.action_dim = action_spec.shape[0]
        self.action_bound = (action_spec.minimum, action_spec.maximum)

        if params is None:
            self.params = Hyperparameters()
        else:
            self.params = params
        # self.learning_rate = learning_rate
        # self.gamma = gamma
        # self.epsilon = epsilon

        # Initialize Q-Network
        self.model = tf.keras.Sequential(
            [
                layers.Dense(32, input_shape=(state_dim,), activation="relu"),
                layers.Dense(32, activation="relu"),
                layers.Dense(2),  # Discrete action dim self.action_dim),
            ]
        )
        self.optimizer = tf.keras.optimizers.legacy.Adam(
            learning_rate=float(self.params.learning_rate)
        )

        # Action noise
        # self.noise = OrnsteinUhlenbeckActionNoise(
        #    mean=np.zeros(action_dim),
        #    std_deviation=self.params.noise_std * np.ones(action_dim),
        # )

        # Add an attribute to hold loss stats
        self.stats = Stats()

    def _flatten_state(self, state):
        return tf.concat([state[key] for key in sorted(state.keys())], axis=0)

    # @tf.function
    def get_action(self, state, epsilon=0):
        flattened_state = self._flatten_state(state)
        flattened_state = tf.reshape(flattened_state, (1, -1))
        q_values = self.model(flattened_state, training=False)

        # Choose a random action with probability epsilon, else choose the action with the highest Q-value
        if tf.random.uniform(()) < epsilon:
            action = tf.random.uniform(
                shape=(self.action_dim,),
                minval=self.action_bound[0],
                maxval=self.action_bound[1],
                dtype=tf.int32,
            )
        else:
            action = (
                tf.constant([1])
                if tf.argmax(q_values[0]).numpy() == 0
                else tf.constant([-1])
            )

        return action

    # @tf.function
    # def get_action(self, state, noise=True):
    #    flattened_state = self._flatten_state(state)
    #    flattened_state = tf.reshape(flattened_state, (1, -1))
    #    action = self.model(flattened_state, training=False)
    #    if noise:
    #        action += tf.cast(self.noise(), tf.float32)
    #    return tf.clip_by_value(action, -self.action_bound, self.action_bound)

    # state = self._flatten_state(state)
    # state = tf.expand_dims(state, axis=0)  # add batch dimension
    # if noise:
    #    q_values = self.model(state, training=False) + tf.cast(
    #        self.noise(), tf.float32
    #    )
    # else:
    #    q_values = self.model(state, training=False)

    # action = np.argmax(q_values[0])
    # print(f"q_values: ${q_values} ${q_values.shape}")
    # print(f"Action: ${action} ${action.shape}")
    # print(f"Action bound: ${self.action_bound} ${self.action_bound.shape}")
    # return action  # tf.clip_by_value(action, -self.action_bound, self.action_bound)

    def train(self, state, action, next_state, reward, done):
        state = self._flatten_state(state)
        next_state = self._flatten_state(next_state)

        loss, q_values = self.train_step(state, action, next_state, reward, done)

        # Save the stats
        self.stats.episode_rewards.append(reward)
        self.stats.losses.append(loss.numpy())
        self.stats.actions.append(action.numpy())
        self.stats.qvalues_0.append(q_values[0][0].numpy())
        self.stats.qvalues_1.append(q_values[0][1].numpy())

        return loss

    @tf.function
    def train_step(self, state, action, next_state, reward, done):
        with tf.GradientTape() as tape:
            state = tf.expand_dims(state, axis=0)  # add batch dimension
            next_state = tf.expand_dims(next_state, axis=0)  # add batch dimension

            # Calculate Q(s, a)
            q_values = self.model(state)
            q_sa = tf.gather(q_values[0], action)

            # Calculate max_a' Q(s', a')
            next_q_values = self.model(next_state)
            q_s_a_prime = tf.reduce_max(next_q_values[0])

            # Convert reward, done, and self.gamma to same type
            reward = tf.cast(reward, tf.float32)
            done = tf.cast(done, tf.float32)
            gamma = tf.cast(self.params.gamma, tf.float32)

            # Calculate target for Q(s, a)
            q_target_sa = reward + (1.0 - done) * gamma * q_s_a_prime

            # Calculate loss
            loss = tf.square(q_sa - q_target_sa)

        # Optimize
        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        return loss, q_values

    def write_summaries(self, episode_num):
        self.stats.write_summaries(episode_num)
        self.stats.clear()
