from collections import deque
import random
from typing import Tuple
import tensorflow as tf
from tensorflow.python.framework.ops import Tensor


class ReplayBuffer:
    def __init__(self, size: int, num_tensors: int):
        self.size = size
        self.num_tensors = num_tensors
        self.experiences = deque()

    def store(self, experience: Tuple[Tensor, ...]) -> None:
        """Stores a tuple of tensors of shape (1, *tensor_shape)"""
        assert (
            len(experience) == self.num_tensors
        ), "Number of tensors in the experience does not match num_tensors"
        self.experiences.append(experience)
        while len(self.experiences) > self.size:
            self.experiences.popleft()

    def sample(self, batch_size: int) -> Tuple[Tensor, ...]:
        """Returns a tuple of tensors of shape (batch_size, *tensor_shape)"""
        samples = random.sample(self.experiences, batch_size)
        return tuple(
            tf.stack([s[i] for s in samples], axis=0) for i in range(self.num_tensors)
        )

    def __len__(self):
        return len(self.experiences)
