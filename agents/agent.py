# agends/agent.py
import abc


class BaseAgent(abc.ABC):
    @abc.abstractmethod
    def get_action(self, state, noise=True):
        """
        Given a state, return the agent's action.
        """
        pass

    @abc.abstractmethod
    def train(self, state, action, next_state, reward, done):
        """
        Train the agent with the provided state, action, next state, reward, and done flag.
        """
        pass

    @abc.abstractmethod
    def write_summaries(self, episode_num):
        """
        Write summaries of the agent's performance for the given episode number.
        """
        pass
