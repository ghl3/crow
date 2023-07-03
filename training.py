import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from datetime import datetime
import tensorflow as tf
from dm_control import suite

import agents.qlearning
import agents.ddpg

import display
import timer
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--log_dir", help="log directory", type=str, default="logs")
parser.add_argument(
    "--num_episodes", help="number of episodes to train for", type=int, default=500
)
parser.add_argument(
    "--use_gpu",
    help="whether to use the GPU",
    default=False,
    action="store_true",
)
args = parser.parse_args()


def train(env, agent, num_episodes=500, log_dir="logs"):
    # Set a directory to store the TensorBoard logs
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = os.path.join(log_dir, current_time)
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    print(train_log_dir)

    # Training loop
    # EPISODES = 500

    # Epsilon annealing settings
    epsilon_start = 1.0
    epsilon_end = 0.1
    annealing_episodes = 400

    epsilon = epsilon_start
    epsilon_decay_value = (epsilon_start - epsilon_end) / annealing_episodes

    tmr = timer.Timer()

    for episode in tqdm(range(num_episodes)):
        timestep = env.reset()
        state = None
        next_state = {
            key: np.array(value) for key, value in timestep.observation.items()
        }
        episode_reward = 0
        episode_step = 0

        # Run the episode
        while not timestep.last():
            tmr.checkpoint("get_action")
            state = next_state
            action = agent.get_action(state, epsilon=epsilon)

            # Take the action and transition to the next state
            tmr.checkpoint("take_action")
            timestep = env.step(action)
            next_state = {
                key: np.array(value) for key, value in timestep.observation.items()
            }
            reward = timestep.reward
            done = timestep.last()

            tmr.checkpoint("train")
            agent.train(
                state=state,
                action=action,
                next_state=next_state,
                reward=reward,
                done=done,
            )

            tmr.checkpoint("increment")
            episode_reward += reward
            episode_step += 1

        # Anneal epsilon at the end of the episode
        if epsilon > epsilon_end:
            epsilon -= epsilon_decay_value

        with train_summary_writer.as_default():
            tmr.checkpoint("write_summaries")
            agent.write_summaries(episode)
            tf.summary.scalar("episode_reward", episode_reward, step=episode)
            tf.summary.scalar("num_episode_steps", episode_step, step=episode)
            tf.summary.scalar("epsilon", epsilon, step=episode)

            if episode % 10 == 0:
                # This runs a new epsisode with epsilon=0 and records it.
                display.save_episode_as_video(
                    env, agent, os.path.join(train_log_dir, f"movie_{episode}.mp4")
                )


def main():
    if not args.use_gpu:
        # Disable GPU
        print("use_gpu not set.  Disabling GPU")
        tf.config.set_visible_devices([], "GPU")

    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = os.path.join(log_dir, current_time)

    # Create the Cartpole environment
    env = suite.load(domain_name="cartpole", task_name="swingup")

    # Initialize the agent
    agent = agents.ddpg.DDPGAgent(env.observation_spec(), env.action_spec())
    # agent = agents.qlearning.QLearningAgent(env.observation_spec(), env.action_spec())

    # Train the agent
    train(env, agent, log_dir=train_log_dir, num_episodes=args.num_episodes)


if __name__ == "__main__":
    main()
