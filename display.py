import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import numpy as np
from PIL import Image


def display_video(frames, framerate=30):
    height, width, _ = frames[0].shape
    dpi = 70
    orig_backend = matplotlib.get_backend()
    matplotlib.use("Agg")  # Switch to headless 'Agg' to inhibit figure rendering.
    fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
    matplotlib.use(orig_backend)  # Switch back to the original backend.
    ax.set_axis_off()
    ax.set_aspect("equal")
    ax.set_position([0, 0, 1, 1])
    im = ax.imshow(frames[0])

    def update(frame):
        im.set_data(frame)
        return [im]

    interval = 1000 / framerate
    anim = animation.FuncAnimation(
        fig=fig, func=update, frames=frames, interval=interval, blit=True, repeat=False
    )
    return HTML(anim.to_html5_video())


def display_3d(trajectory_dict):
    """
    trajectory_dict: Each
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for name, df in trajectory_dict.items():
        x, y, z = df["x"].iloc[:50], df["y"].iloc[:50], df["z"].iloc[:50]
        ax.scatter(x, y, z, label=name)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.legend()

    plt.show()


def save_episode_as_video(env, agent, video_filename):
    timestep = env.reset()
    done = False

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(video_filename, fourcc, 20.0, (640, 480))

    while not done:
        # state = np.array(timestep.observation)
        state = {key: np.array(value) for key, value in timestep.observation.items()}
        action = agent.get_action(state, epsilon=0)
        timestep = env.step(action)
        done = timestep.last()

        # Render the environment
        # image = env.render(mode='rgb_array')
        frame = env.physics.render(camera_id=0, width=640, height=480)

        # if video_writer is None:
        #    # Initialize the video writer if it hasn't been initialized
        #    height, width, _ = image.shape
        #    video_writer = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    video.release()
