import tensorflow as tf
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.layers import Dense, Input
from tensorflow.python.keras.optimizers import *
import numpy as np

class DLModel:
    """
    The main model for deep learning simulation and updates.
    """
    def __init__(self, state_sizes, action_sizes, name, arglist):
        self.state_sizes = state_sizes
        self.action_sizes = action_sizes
        self.name = name
        self.model = self.build_and_compile_model()
        self.targetModel = self.build_and_compile_model()
        self.alpha = arglist.learning_rate
        self.num_nodes = arglist.num_nodes
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    def load_model_trained(self):
        """
        Load the model to be trained, currently archived.
        """
        self.model = load_model(self.name)
    
    def build_and_compile_model(self):
        """
        Build the models required for the deep Q-learning.
        Return:
            The model created and compiled
        """
        x = Input(shape=(self.state_sizes,))
        x1 = Dense(self.num_nodes, activation='relu')(x)
        x2 = Dense(self.num_nodes, activation='relu')(x1)    
        z =  Dense(self.action_sizes, activation='softmax')(x2)
        model = Model(inputs=x, outputs=z)
        model.compile(loss="MeanSquaredError", optimizer="adam")
        
        return model
    
    def update_target(self):
        """
        Update the target model weights for later training.
        """
        self.targetModel.set_weights(self.model.get_weights())
    
    def predict(self, state, target=False):
        """
        Predict the probabilities based on the state.

        Args:
            state: The representation of the environment
            target: Whether to update with target, used in y_update
        Return:
            The predict result
        """
        if not target:
            return self.model.predict(state)
        else:
            return self.targetModel.predict(state)

    def non_test_weight_loading(self):
        """
        To load the weights needed for final testing.
        """
        self.model.load_weights(self.name)
    
    def train_model(self, X, y, epochs=10, verbose=0):
        """
        Fit the model based on epochs and verbose requirements,
        used during the y_update.

        Args:
            X: The values for training
            y: The "correct" values in the reinforcement learning framework
            epochs: Number of times the training continues
            verbose: The logging level
        """
        self.model.fit(X, y, batch_size=len(X), epochs=epochs, verbose=verbose)

    def max_Q_action(self, state, legalActions, target=False):
        """
        Return the action that has the maximum Q-value, used during epsilon-greedy
        algorithm. As not all actions are legal, eliminations are required.
        """
        actions = self.predict(state.reshape(1, self.state_sizes), target=target)

        finalList = actions.flatten()
        maxIndex = 1000
        maxValue = float("-inf")

        # Remove actions that are not legal
        for i in range(len(finalList)):
            if i not in legalActions:
                continue
            else:
                if finalList[i] > maxValue:
                    maxIndex = i
        return maxIndex

    def save_model(self):
        """
        Save the model to the filename
        """
        self.model.save(self.name)
