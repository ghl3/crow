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
from timer import Timer
from transition_history import TransitionHistory
from transition import Transition


@dataclass
class Stats:
    actor_losses: list = field(default_factory=list)
    critic_losses: list = field(default_factory=list)
    episode_rewards: list = field(default_factory=list)
    critic_nmses: list = field(default_factory=list)

    def clear(self):
        self.actor_losses.clear()
        self.critic_losses.clear()
        self.episode_rewards.clear()
        self.critic_nmses.clear()

    def write_summaries(self, episode_num):
        tf.summary.scalar(
            "avg_actor_loss", np.mean(self.actor_losses), step=episode_num
        )
        tf.summary.scalar(
            "avg_critic_loss", np.mean(self.critic_losses), step=episode_num
        )
        tf.summary.scalar(
            "avg_critic_nmse", np.mean(self.critic_nmses), step=episode_num
        )
        tf.summary.scalar(
            "episode_reward", np.sum(self.episode_rewards), step=episode_num
        )


# Actor network or Policy network
# The actor takes a state and returns an action.
# It determines what the next action should be.
def create_actor(state_dim, action_dim, action_bound):
    inputs = layers.Input(shape=(state_dim,))
    x = layers.Dense(
        10,
        kernel_initializer=HeNormal(),
        bias_initializer=Zeros(),
        kernel_regularizer=l2(1e-5),
    )(inputs)
    x = BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dense(
        5,
        kernel_initializer=HeNormal(),
        bias_initializer=Zeros(),
        kernel_regularizer=l2(1e-5),
    )(x)
    x = BatchNormalization()(x)
    x = layers.ReLU()(x)
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
    x = layers.Dense(
        10,
        kernel_initializer=HeNormal(),
        bias_initializer=Zeros(),
        kernel_regularizer=l2(1e-5),
    )(inputs)
    x = BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dense(
        5,
        kernel_initializer=HeNormal(),
        bias_initializer=Zeros(),
        kernel_regularizer=l2(1e-5),
    )(x)
    x = BatchNormalization()(x)
    x = layers.ReLU()(x)
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


class DDPGAgent(BaseAgent):
    def __init__(self, state_dim, action_dim, action_bound, params):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound

        if params is None:
            self.params = Hyperparameters()
        else:
            self.params = params

        self.memory = TransitionHistory()

        # Create networks
        self.actor = create_actor(state_dim, action_dim, action_bound)
        self.actor_target = create_actor(state_dim, action_dim, action_bound)
        self.critic = create_critic(state_dim, action_dim)
        self.critic_target = create_critic(state_dim, action_dim)

        self.stats = Stats()

        # Action noise
        self.noise = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(action_dim),
            std_deviation=self.params.noise_std * np.ones(action_dim),
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

        self._timer = Timer()

    @tf.function
    def get_action(self, state, noise=True):
        flattened_state = self._flatten_state(state)
        flattened_state = tf.reshape(flattened_state, (1, -1))
        action = self.actor(flattened_state)
        if noise:
            action += self.noise()
        return tf.clip_by_value(action, -self.action_bound, self.action_bound)

    def train(self, state, action, next_state, reward, done):
        self._timer.checkpoint("transition")
        transition = Transition(
            state=self._flatten_state(state),
            action=action,
            next_state=self._flatten_state(next_state),
            reward=reward,
            done=done,
        )
        self.memory.add_transition(transition)
        if len(self.memory) < self.params.batch_size:
            return

        self._timer.checkpoint("get_batch")
        batch = self.memory.get_batch(self.params.batch_size)

        self._timer.checkpoint("train_batch")
        training_metrics = self._train_batch(
            states=batch.states,
            actions=batch.actions,
            next_states=batch.next_states,
            rewards=batch.rewards,
            dones=batch.dones,
        )

        self._timer.checkpoint("training_metrics")
        self.stats.episode_rewards.append(reward)
        self.stats.actor_losses.append(training_metrics["actor_loss"].numpy())
        self.stats.critic_losses.append(training_metrics["critic_loss"].numpy())
        self.stats.critic_nmses.append(training_metrics["critic_nmse"].numpy())

        self._timer.stop()

    @tf.function
    def _train_batch(self, states, actions, next_states, rewards, dones):
        #
        # Calculate the 'target' q-value using target netowrks
        #
        # Given the next state, pick a next action
        target_actions = self.actor_target(next_states)
        # Then, calculate the q-value for that (next_state, next_action)
        target_q_values = self.critic_target([next_states, target_actions])
        # And add it to the existing reward to get the full reward
        target_q_values = (
            rewards
            + (tf.cast(1.0, dtype=tf.float32) - tf.cast(dones, dtype=tf.float32))
            * tf.cast(self.params.gamma, dtype=tf.float32)
            * target_q_values
        )

        #
        # Update the Critic
        #
        with tf.GradientTape() as tape:
            # Then, we calculate the q-value using the normal critic
            q_values = self.critic([states, actions])
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
            actor_actions = self.actor(states)
            # Use the critic to determine the q-value for that action
            q_values = self.critic([states, actor_actions])
            # And derive a loss from that q-value
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
        return tf.concat([state[key] for key in sorted(state.keys())], axis=0)
