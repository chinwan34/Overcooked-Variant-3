import random
from utils.UER import *
import numpy as np
from utils.dlmodel import *

class DQNAgent:
    """
    The agent that holds the information and models for deep learning update, 
    while executing actions for updates.
    """
    def __init__(self, arglist, st_size, action_size, name, color, role, agent_index, dlmodel_name, gamma=0.95, epsilon=0.8):
        self.name = name
        self.st_size = st_size
        self.action_size = action_size
        self.color = color
        self.role = role
        self.alpha = arglist.learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay_rate = 0.99
        self.epsilon_minimum = 0
        self.agent_index = agent_index
        self.dlmodel_name = dlmodel_name
        self.maxCapacity = arglist.maxCapacity
        self.batchSize = arglist.batch_size
        self.frequency = arglist.update_frequency
        self.current = 0

        self.dlmodel = DLModel(self.st_size, self.action_size, self.dlmodel_name, arglist)
        self.memory = UER_memory(self.maxCapacity)

        # Actions that are legal
        self.plausibleActions = []

        self.maxEpsilon = 0.8
        self.minEpsilon = 0.02
        self.maxExplorationSteps = 20000000
    
    def legal_actons(self, legalActions):
        """
        Add actions that are legal in the environment, 
        utilizing for the epsilon-greedy algorithm.

        Args:
            legalActions: The actions that are legal in the environment
        """
        self.plausibleActions.clear()
        actions_available = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for action in legalActions:
            self.plausibleActions.append(actions_available.index(action))
        print(self.plausibleActions)

    def epsilon_greedy(self, state):
        """
        Epsilon-greedy on action selection, allow exploration-exploitation
        trade-off.

        Args:
            state: The current state of the environment
        Return:
            The maximum action number for update
        """
        value = random.randint(0, 1)
        if value < self.epsilon:
            return random.choice(self.plausibleActions)
        else:
            return self.dlmodel.max_Q_action(state, self.plausibleActions)

    def epsilon_decay(self):
        """
        The epsilon-decay function to gradually reduce epsilon through time, 
        reducing exploration possibilities.
        """
        self.current += 1
        if self.current < self.maxExplorationSteps:
            self.epsilon = (self.maxEpsilon - self.minEpsilon) * ((self.maxExplorationSteps - self.current)/self.maxExplorationSteps) + self.minEpsilon
    
    def update_target(self):
        """
        To update the target when frequency matches
        """
        if self.current % self.frequency == 0:
            self.dlmodel.update_target()
    
    def update_result(self):
        """
        To train the model based on the update of DQN algorithm.
        """
        batch_used = self.memory.uniform_sample(self.batchSize)
        x, y, errors = self.y_i_update(batch_used)
        self.dlmodel.train_model(x, y)

    def y_i_update(self, batch_used):
        """
        The main update algorithm for training, update 
        the y value based on the batch sampled uniformly from Uniform Experience
        Memory file. 
        """
        current_states = np.array([batch[0] for batch in batch_used])
        next_states = np.array([batch[3] for batch in batch_used])
        predict_current = self.dlmodel.predict(current_states)
        predict_next_target = self.dlmodel.predict(next_states, target=True)

        x = np.zeros((len(batch_used), self.st_size))
        y = np.zeros((len(batch_used), self.action_size))
        errors = np.zeros(len(batch_used))

        for i in range(0, len(batch_used)):
            cState, actionSelected, reward, done = batch_used[i][0], batch_used[i][1][self.name], batch_used[i][2], batch_used[i][4]

            batchSelected = predict_current[i]
            oldVal = batchSelected[actionSelected]

            # Check if it is in the last state
            if done: batchSelected[actionSelected] = reward
            else: batchSelected[actionSelected] = reward + self.gamma * np.max(predict_next_target[i])
        
            x[i] = cState
            y[i] = batchSelected
            errors[i] = np.abs(batchSelected[actionSelected] - oldVal)

        return x, y, errors

    def load_model_trained(self):
        """
        Load the model weights that are trained for
        testing purposes.
        """
        self.dlmodel.non_test_weight_loading()
    
    def set_alpha_and_epsilon(self):
        """
        Setting alpha and epsilon to 0 for testing environment,
        currently archived.
        """
        self.alpha = 0
        self.epsilon = 0
    
    def predict(self, state):
        """
        The predict function during testing, return maximum actions
        without possibility of explorations.
        
        Args:
            state: Current state of the environment
        Return:
            The maximum action number
        """
        return self.dlmodel.max_Q_action(state, self.plausibleActions)

    def observeTransition(self, transition):
        """
        Store the transition of the current step to the memory.

        Args;
            transition: The values that should be stored
        """
        self.memory.store_transition(transition)
    

    
    


            

    
