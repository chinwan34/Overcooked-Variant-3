import tensorflow as tf
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.layers import Dense, Input
from tensorflow.python.keras.optimizers import *
import numpy as np

class DLModel:
    def __init__(self, state_sizes, action_sizes, name, arglist):
        self.state_sizes = state_sizes
        self.action_sizes = action_sizes
        self.name = name
        self.alpha = arglist.learning_rate
        self.num_nodes = arglist.num_nodes
        self.model = self.build_and_compile_model()
        self.targetModel = self.build_and_compile_model()
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    def load_model_trained(self):
        self.model = load_model(self.name)
    
    def update_target(self):
        self.targetModel.set_weights(self.model.get_weights())
    
    def build_and_compile_model(self):
        x = Input(shape=(self.state_sizes,))
        x1 = Dense(self.num_nodes, activation='relu')(x)
        x2 = Dense(self.num_nodes, activation='relu')(x1)    
        z =  Dense(self.action_sizes, activation='softmax')(x2)
        model = Model(inputs=x, outputs=z)

        model.compile(loss="MeanSquaredError", optimizer="adam")
        
        return model
    
    def non_test_weight_loading(self):
        self.model.load_weights(self.name)
    
    def train_model(self, X, y, epochs=10, verbose=0):
        self.model.fit(X, y, batch_size=len(X), epochs=epochs, verbose=verbose)

    def predict(self, state, target=False):
        if not target:
            return self.model.predict(state)
        else:
            return self.targetModel.predict(state)
    
    def max_Q_action(self, state, legalActions, target=False):
        actions = self.predict(state.reshape(1, self.state_sizes), target=target)

        finalList = actions.flatten()
        maxIndex = 1000
        maxValue = float("-inf")

        for i in range(len(finalList)):
            if i not in legalActions:
                continue
            else:
                if finalList[i] > maxValue:
                    maxIndex = i
        return maxIndex

        # return np.argmax((actions.flatten()))

    def save_model(self):
        self.model.save(self.name)
