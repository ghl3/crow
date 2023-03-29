from collections import defaultdict
import pandas as pd
import pandas as pd
import numpy as np
from dm_control.utils import transformations


def get_system_state(physics):
    body_names = [
        physics.model.id2name(body_id, "body") for body_id in range(physics.model.nbody)
    ]

    # Get body positions (xpos)
    body_positions = pd.DataFrame(
        physics.named.data.xpos, index=body_names, columns=["x", "y", "z"]
    )

    # Get body orientations (xmat) and convert them to quaternions
    body_orientations = np.reshape(physics.named.data.xmat, (len(body_names), 3, 3))
    body_orientations_quat = [
        transformations.mat_to_quat(mat) for mat in body_orientations
    ]
    body_orientations = pd.DataFrame(
        body_orientations_quat, index=body_names, columns=["qw", "qx", "qy", "qz"]
    )

    # Create a multi-index DataFrame with body positions and orientations
    system_state = pd.concat(
        [body_positions, body_orientations], axis=1, keys=["position", "orientation"]
    )

    # Skip the first body, which is 'World'
    return system_state.iloc[1:]


def states_to_matrix(states):
    return pd.DataFrame(
        [
            state["state"].unstack().reorder_levels([2, 0, 1]).rename(state["time"])
            for state in states
        ]
    )
