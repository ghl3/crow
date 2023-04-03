# DDPG agent
from dataclasses import dataclass
from transition_history import TransitionHistory
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
from transition import Transition


# Actor network or Policy network
# The actor takes a state and returns an action.
# It determines what the next action should be.
def create_actor(state_dim, action_dim, action_bound):
    inputs = layers.Input(shape=(state_dim,))
    x = layers.Dense(40, activation="relu")(inputs)
    x = layers.Dense(30, activation="relu")(x)
    raw_actions = layers.Dense(action_dim, activation="tanh")(x)
    actions = raw_actions * action_bound
    return tf.keras.Model(inputs=inputs, outputs=actions)


# Critic network or Value network or Q-function
# The critic takes a state and an action and returns a q-value.
# The critic observes the action suggested by the actor
# and predicts how good of an action that will be.
def create_critic(state_dim, action_dim):
    state_inputs = layers.Input(shape=(state_dim,))
    action_inputs = layers.Input(shape=(action_dim,))
    inputs = layers.Concatenate()([state_inputs, action_inputs])
    x = layers.Dense(40, activation="relu")(inputs)
    x = layers.Dense(30, activation="relu")(x)
    q_values = layers.Dense(1)(x)
    return tf.keras.Model(inputs=[state_inputs, action_inputs], outputs=q_values)


@dataclass
class Hyperparameters:
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 0.001
    critic_lr: float = 0.002
    noise_std: float = 0.1
    batch_size: int = 256
    memory_capacity: int = int(1e6)


@dataclass
class TrainResult:
    actor_loss: float
    critic_loss: float


class DDPGAgent:
    def __init__(self, state_dim, action_dim, action_bound, params=Hyperparameters()):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.params = params

        self.memory = TransitionHistory()

        # Create networks
        self.actor = create_actor(state_dim, action_dim, action_bound)
        self.actor_target = create_actor(state_dim, action_dim, action_bound)
        self.critic = create_critic(state_dim, action_dim)
        self.critic_target = create_critic(state_dim, action_dim)

        # Initialize target networks
        # self.update_target_networks(tau=1)

        # Optimizers
        self.actor_optimizer = tf.keras.optimizers.legacy.Adam(
            learning_rate=self.params.actor_lr
        )
        self.critic_optimizer = tf.keras.optimizers.legacy.Adam(
            learning_rate=self.params.critic_lr
        )

        self.episode_actor_losses = []
        self.episode_critic_losses = []
        self.episode_rewards = []

    def get_action(self, state, noise=True):
        flattened_state = self._flatten_state(state).reshape((1, -1))
        action = self.actor(flattened_state)
        if noise:
            action += np.random.normal(0, self.params.noise_std, size=action.shape)
        return np.clip(action, -self.action_bound, self.action_bound)

    def train(self, state, action, next_state, reward, done):
        transition = Transition(
            state=self._flatten_state(state),
            action=action,
            next_state=self._flatten_state(next_state),
            reward=reward,
            done=done,
        )
        self.episode_rewards.append(reward)
        self.memory.add_transition(transition)

        if len(self.memory) < self.params.batch_size:
            return

        batch = self.memory.get_batch(self.params.batch_size)

        #
        # Calculate the 'target' q-value using target netowrks
        #

        # Given the next state, pick a next action
        target_actions = self.actor_target(batch.next_states)
        # Then, calculate the q-value for that (next_state, next_action)
        target_q_values = self.critic_target([batch.next_states, target_actions])
        # And add it to the existing reward to get the full reward
        target_q_values = (
            batch.rewards
            + (1 - np.array(batch.dones)) * self.params.gamma * target_q_values
        )

        #
        # Update the Critic
        #
        with tf.GradientTape() as tape:
            # Then, we calculate the q-value using the normal critic
            q_values = self.critic([batch.states, np.array(batch.actions)])
            # and we compare it to our target q-value and use that to derive the loss.
            critic_loss = tf.reduce_mean(tf.square(target_q_values - q_values))
            critic_grads = tape.gradient(critic_loss, self.critic.trainable_weights)
        self.critic_optimizer.apply_gradients(
            zip(critic_grads, self.critic.trainable_weights)
        )

        #
        # Update the Actor
        #
        with tf.GradientTape() as tape:
            # Get the action for the given state
            actions = self.actor(batch.states)
            # Use the critic to determine the q-value for that action
            q_values = self.critic([batch.states, actions])
            # And derive a loss from that q-value
            actor_loss = -tf.reduce_mean(q_values)
            actor_grads = tape.gradient(actor_loss, self.actor.trainable_weights)
        self.actor_optimizer.apply_gradients(
            zip(actor_grads, self.actor.trainable_weights)
        )

        # Update the target networks
        self._update_target_networks()

        # Accumulate the losses for summaries
        self.episode_actor_losses.append(actor_loss.numpy())
        # Convert the loss to a RMS and scale by the target q values to make it a percentage
        # This avoids the loss increasing as q increases, which is a bit odd for Tensorboard
        self.episode_critic_losses.append(
            np.sqrt(critic_loss.numpy()) / target_q_values.numpy()
        )

    def write_summaries(self, episode_num):
        episode_reward = np.sum(self.episode_rewards)
        avg_actor_loss = np.mean(self.episode_actor_losses)
        avg_critic_loss = np.mean(self.episode_critic_losses)
        tf.summary.scalar("avg_actor_loss", avg_actor_loss, step=episode_num)
        tf.summary.scalar("avg_critic_loss", avg_critic_loss, step=episode_num)
        tf.summary.scalar("episode_reward", episode_reward, step=episode_num)
        self.episode_actor_losses = []
        self.episode_critic_losses = []
        self.episode_rewards = []

    def _update_target_networks(self):
        tau = self.params.tau
        for target, source in zip(self.actor_target.weights, self.actor.weights):
            target.assign(tau * source + (1 - tau) * target)

        for target, source in zip(self.critic_target.weights, self.critic.weights):
            target.assign(tau * source + (1 - tau) * target)

    def _flatten_state(self, state):
        return np.concatenate([state[key] for key in sorted(state.keys())])
