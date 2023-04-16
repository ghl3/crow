# simulate.py

import numpy as np
import dm_control.mujoco as mujoco
from dm_control.mujoco.wrapper.mjbindings import enums

from process import get_system_state


def simulate(physics, duration=10, frames_per_sec=60):
    # Track when we wite frames
    last_frame_time = None
    seconds_per_frame = 1.0 / frames_per_sec

    # Simulate and display video.
    states = []
    frames = []

    # Visualize the joint axis
    scene_option = mujoco.wrapper.core.MjvOption()
    # scene_option.flags[enums.mjtVisFlag.mjVIS_JOINT] = True

    while physics.time() < duration:
        time = physics.time()

        # Set the values
        # physics.named.data.ctrl["elbow_motor"] = 0.1

        # Get the state of the model at the start of the time window
        states.append({"time": physics.time(), "state": get_system_state(physics)})

        if last_frame_time is None or time > last_frame_time + seconds_per_frame:
            pixels = physics.render(scene_option=scene_option)
            frames.append(pixels)
            last_frame_time = time

        # Jump to the next state
        physics.step()

    return states, frames
