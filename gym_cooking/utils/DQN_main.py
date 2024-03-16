import numpy as np
import random
from utils.DQNagent import DQNAgent

class mainAlgorithm:
    def __init__(self, environment, arglist):
        self.arglist = arglist
        self.environment = environment
        self.num_training = self.arglist.number_training
        self.max_timestep = self.arglist.max_timestep
        self.filling_step = 15
        self.replay_step = self.arglist.replay

    def run(self, agents):
        all_step = 0
        rewards = []
        time_steps = []
        maxScore = 0
        for episode in range(self.num_training):
            state = self.environment.reset()
            
            # May not be needed
            # state = np.array(state)
            # state = state.ravel()
        
            done = False
            step = 0
            rewardTotal = 0

            while not done and step < self.max_timestep:
                action_dict = []
                for agent in agents:
                    action_dict.append(agent.epsilon_greedy(state))
                
                next_state, reward, done, info = self.environment.step(action_dict)
                next_state = np.array(next_state)
                # next_state = next_state.ravel()

                for agent in agents:
                    agent.observeTransition((state, action_dict, reward, next_state, done))
                    if all_step >= self.filling_step:
                        agent.epsilon_decay()
                        if step % self.replay_step == 0:
                            agent.replay()
                        
                all_step += 1
                step += 1
                state = next_state
                rewardTotal += reward

                # Render here maybe
            
            rewards.append(rewardTotal)
            time_steps.append(step)

            if episode % 100 == 0:
                if all_step >= self.replay_step:
                    if rewardTotal > maxScore:
                        for agent in agents:
                            self.agent.dlmodel.save_model()               
                        maxScore = rewardTotal

            print("Score:{s} with Steps:{t}, Goal:{g}".format(s=rewardTotal, t=step, g=done))


