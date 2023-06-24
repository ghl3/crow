# agents/agent.py
import abc


class BaseAgent(abc.ABC):
    @abc.abstractmethod
    def get_action(self, state, epsilon=0.0):
        """
        Given a state, return the agent's action.
        - state: the state the agent is in
        - epsilon: the probability of taking a random action

        The return value should be a numpy array of shape (action_dim,).
        """
        pass

    @abc.abstractmethod
    def train(self, state, action, next_state, reward, done):
        """
        Train the agent with the provided state, action, next state, reward, and done flag.
        - state: the state the agent was in when it took the action
        - action: the action the agent took
        - next_state: the state the agent transitioned to after taking the action
        - reward: the reward the agent received for taking the action
        - done: whether the agent has reached a terminal state
        """
        pass

    @abc.abstractmethod
    def write_summaries(self, episode_num):
        """
        Write summaries of the agent's performance for the given episode number.
        - episode_num: the episode number
        """
        pass
