from collections import deque
import random

class UER_memory:
    """
    The memory storage space for all seen transitions of a DQNAgent.
    """
    def __init__(self, maxCapacity):
        self.maxCapacity = maxCapacity
        self.memory = deque(maxlen = self.maxCapacity)
    
    def uniform_sample(self, size):
        """
        To sample the batch for updates uniformly
        Args:
            size: Size of the batch
        Return:
            The batch of samples
        """
        if size >= len(self.memory):
            return random.sample(self.memory, len(self.memory))
        else:
            return random.sample(self.memory, size)
    
    def store_transition(self, transition):
        """
        Store the transition seen into the memory
        Args:
            transition: The transition tuple
        """
        self.memory.append(transition)
    
    
