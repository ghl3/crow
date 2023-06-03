# agents/qlearning.py


import random
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.experimental.numpy as np


class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = []
        self.max_size = max_size

    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)


class QLearningAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        learning_rate=0.01,
        gamma=0.99,
        batch_size=64,
        buffer_size=10000,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer = ReplayBuffer(buffer_size)
        self.optimizer = tf.keras.optimizers.legacy.Adam(
            learning_rate=self.learning_rate
        )
        self.model = self.create_q_model()

    def create_q_model(self):
        model = tf.keras.models.Sequential(
            [
                layers.Dense(32, activation="relu", input_shape=(self.state_dim,)),
                layers.Dense(32, activation="relu"),
                layers.Dense(self.action_dim, activation="linear"),
            ]
        )
        return model

    def get_action(self, state, epsilon=0):
        if np.random.rand() <= epsilon:
            return np.random.choice(self.action_dim)  # Random action (explore)
        else:
            q_values = self.model(state[None, :], training=False)
            return np.argmax(q_values[0])  # Best action (exploit)

    def remember(self, state, action, reward, next_state, done):
        self.buffer.add(state, action, reward, next_state, done)

    @tf.function
    def train(self):
        if len(self.buffer.buffer) < self.batch_size:
            return

        batch = self.buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        states = states.reshape(self.batch_size, self.state_dim)
        next_states = next_states.reshape(self.batch_size, self.state_dim)

        with tf.GradientTape() as tape:
            q_values = self.model(states, training=True)
            q_next = tf.stop_gradient(self.model(next_states, training=False))
            # Q-learning update
            target_q_values = q_values.numpy()
            max_q_next = np.amax(q_next, axis=1)
            for i, action in enumerate(actions):
                if dones[i]:
                    target_q_values[i][action] = rewards[i]
                else:
                    target_q_values[i][action] = rewards[i] + self.gamma * max_q_next[i]
            target_q_values = tf.convert_to_tensor(target_q_values)
            loss = tf.reduce_mean(tf.square(target_q_values - q_values))

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
