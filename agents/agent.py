import abc


class BaseAgent(abc.ABC):
    @abc.abstractmethod
    def get_action(self, state, noise=True):
        pass

    @abc.abstractmethod
    def train(self, state, action, next_state, reward, done):
        pass

    @abc.abstractmethod
    def write_summaries(self, episode_num):
        pass
