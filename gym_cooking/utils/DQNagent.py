import random
from utils.UER import *
import numpy as np
from utils.dlmodel import *

class DQNAgent:
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

        self.plausibleActions = []

        self.maxEpsilon = 0.8
        self.minEpsilon = 0.02
        self.maxExplorationSteps = 20000000
    
    def legal_actions(self, legalActions):
        self.plausibleActions.clear()
        actions_available = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for action in legalActions:
            self.plausibleActions.append(actions_available.index(action))
        print(self.plausibleActions)

    def epsilon_greedy(self, state):
        """
        Epsilon-greedy on action selection
        Need to implement legal action part 
        """
        value = random.randint(0, 1)
        if value < self.epsilon:
            return random.choice(self.plausibleActions)
        else:
            return self.dlmodel.max_Q_action(state, self.plausibleActions)

    def epsilon_decay(self):
        self.current += 1
        if self.current < self.maxExplorationSteps:
            self.epsilon = (self.maxEpsilon - self.minEpsilon) * ((self.maxExplorationSteps - self.current)/self.maxExplorationSteps) + self.minEpsilon
    
    def update_target(self):
        if self.current % self.frequency == 0:
            self.dlmodel.update_target()

    def y_i_update(self, batch_used):
        current_states = np.array([batch[0] for batch in batch_used])
        next_states = np.array([batch[3] for batch in batch_used])
        predict_current = self.dlmodel.predict(current_states)
        predict_next_target = self.dlmodel.predict(next_states, target=True)

        x = np.zeros((len(batch_used), self.st_size))
        y = np.zeros((len(batch_used), self.action_size))
        errors = np.zeros(len(batch_used))

        for i in range(0, len(batch_used)):
            cState, actionSelected, reward, done = batch_used[i][0], batch_used[i][1][self.name], batch_used[i][2], batch_used[i][4]

            t = predict_current[i]
            oldVal = t[actionSelected]
            if done: t[actionSelected] = reward
            else: t[actionSelected] = reward + self.gamma * np.max(predict_next_target[i])
        
            x[i] = cState
            y[i] = t
            errors[i] = np.abs(t[actionSelected] - oldVal)

        return x, y, errors

    def update_result(self):
        batch_used = self.memory.uniform_sample(self.batchSize)
        x, y, errors = self.y_i_update(batch_used)
        self.dlmodel.train_model(x, y)
    
    def set_alpha_and_epsilon(self):
        self.alpha = 0
        self.epsilon = 0
    
    def predict(self, state):
        return self.dlmodel.max_Q_action(state, self.plausibleActions)

    def observeTransition(self, transition):
        self.memory.store_transition(transition)
    
    def load_model_trained(self):
        self.dlmodel.non_test_weight_loading()

    
    


            

    
