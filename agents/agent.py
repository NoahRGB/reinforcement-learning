from abc import ABC, abstractmethod


class Agent(ABC):

    @abstractmethod
    def run_policy(self, s, t):
        """
        this method should return an action to perform on
        the given state s
        """
        ...

    @abstractmethod
    def update(self, s, sprime, a, r, done):
        """
        this method should perform any necessary learning
        updates based on the given s, s', a, r sequence

        it should return a bool denoting whether the agent
        wants to stop learning
        (perhaps based on a timeout)
        """
        ...

    @abstractmethod
    def initialise(self, state_space_size, action_space_size, start_state, resume=False):
        """
        this method should initialise all relevant structures
        associated with the agent's logic so a new episode can
        begin
        """
        ...

    @abstractmethod
    def finish_episode(self):
        """
        this method should execute the relevant book keeping
        or learning that needs to be done at the end of an
        episode
        """
        ...

    @abstractmethod
    def get_supported_state_spaces(self):
        ...

    @abstractmethod
    def get_supported_action_spaces(self):
        ...
