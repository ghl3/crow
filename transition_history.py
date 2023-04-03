from dataclasses import dataclass
from typing import Dict
import numpy as np
import random


@dataclass
class Transitions:
    states: np.ndarray
    actions: np.ndarray
    next_states: np.ndarray
    rewards: float
    dones: bool


class TransitionHistory:
    def __init__(self, capacity=1e6):
        self.capacity = capacity
        self.transitions = []

    def __len__(self):
        return len(self.transitions)

    def add_transition(self, timestep):
        self.transitions.append(timestep)
        if len(self.transitions) > self.capacity:
            self.transitions.pop(0)

    def get_batch(self, batch_size):
        batch = random.sample(self.transitions, batch_size)

        states = np.concatenate([b.state for b in batch]).reshape((batch_size, -1))
        actions = np.concatenate([b.action for b in batch]).reshape((batch_size, -1))
        next_states = np.concatenate([b.next_state for b in batch]).reshape(
            (batch_size, -1)
        )
        rewards = np.array([b.reward for b in batch]).reshape((batch_size, -1))
        dones = np.array([b.done for b in batch]).reshape((batch_size, -1))

        return Transitions(
            states=states,
            actions=actions,
            next_states=next_states,
            rewards=rewards,
            dones=dones,
        )
