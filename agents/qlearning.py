# agents/qlearning.py
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.experimental.numpy as np


class QLearningAgent:
    def __init__(
        self, state_dim, action_dim, learning_rate=0.01, gamma=0.99, epsilon=0.1
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon

        # Initialize Q-Network
        self.model = tf.keras.Sequential(
            [
                layers.Dense(32, input_shape=(state_dim,), activation="relu"),
                layers.Dense(32, activation="relu"),
                layers.Dense(action_dim),
            ]
        )
        self.optimizer = tf.keras.optimizers.legacy.Adam(
            learning_rate=float(learning_rate)
        )

        # Add an attribute to hold loss stats
        self.loss_stats = []

    def _flatten_state(self, state):
        return tf.concat([state[key] for key in sorted(state.keys())], axis=0)

    @tf.function
    def get_action(self, state, noise=False):
        state = self._flatten_state(state)
        state = tf.expand_dims(state, axis=0)  # add batch dimension
        # epsilon = self.epsilon if epsilon is None else epsilon
        # if np.random.rand() < epsilon:
        #    return np.random.choice(self.action_dim)  # Random action (explore)
        # else:
        q_values = self.model(state, training=False)
        return np.argmax(q_values[0])

    def train(self, state, action, next_state, reward, done):
        state = self._flatten_state(state)
        next_state = self._flatten_state(next_state)

        loss = self.train_step(state, action, next_state, reward, done)

        # Save the loss stats
        self.loss_stats.append(loss.numpy())

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
            gamma = tf.cast(self.gamma, tf.float32)

            # Calculate target for Q(s, a)
            q_target_sa = reward + (1.0 - done) * gamma * q_s_a_prime

            # Calculate loss
            loss = tf.square(q_sa - q_target_sa)

        # Optimize
        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        # Save the loss stats
        # elf.loss_stats.append(loss.numpy())

        return loss

    def write_summaries(self, episode):
        avg_loss = np.mean(self.loss_stats)
        tf.summary.scalar("avg_loss", avg_loss, step=episode)
        # Clear stats
        self.loss_stats.clear()
