# transition.py
from dataclasses import dataclass
from typing import Dict
import numpy as np
import random


@dataclass
class Transition:
    state: np.ndarray
    action: np.ndarray
    next_state: np.ndarray
    reward: np.float32
    done: bool
