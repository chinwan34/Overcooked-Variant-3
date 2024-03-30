import numpy as np
import random
from utils.DQNagent import DQNAgent

class mainAlgorithm:
    def __init__(self, environment, arglist):
        self.arglist = arglist
        self.environment = environment
        self.num_training = self.arglist.number_training
        self.max_timestep = self.arglist.max_num_timesteps
        self.filling_step = 0
        self.replay_step = self.arglist.replay
        self.final_episodes = 5

    def run(self, agents):
        all_step = 0
        rewards = []
        previous_reward = 0
        time_steps = []
        maxScore = float("-inf")
        for episode in range(self.num_training):
            print("EPISODE------------", episode, "-----------EPISODE")
            state = self.environment.reset()
        
            done = False
            step = 0
            rewardTotal = 0
            state = np.array(state)
            state = state.ravel()

            while not done and step < self.max_timestep:
                action_dict = {}
                for agent in agents:
                    legalActions = self.environment.legal_actions(agent.name)
                    agent.legal_actions(legalActions)

                    action = agent.epsilon_greedy(state)
                    action_dict[agent.name] = action
                
                next_state, reward, doneUsed, info = self.environment.dqn_step(action_dict)
                next_state = np.array(next_state)
                next_state = next_state.ravel()

                for agent in agents:
                    agent.observeTransition((state, action_dict, reward, next_state, doneUsed))
                    agent.epsilon_decay()
                    if step % self.replay_step == 0:
                        agent.update_result()
                    agent.update_target()
                
                rewardTotal += reward
                done = doneUsed
                all_step += 1
                step += 1
                state = next_state
            
            rewards.append(rewardTotal)
            time_steps.append(step)

            if episode % 10 == 0:
                if rewardTotal > maxScore:
                    for agent in agents:
                        print("Got in episode for updates")
                        agent.dlmodel.save_model()               
                    maxScore = rewardTotal
            print("Score:{s} with Steps:{t}, Goal:{g}".format(s=rewardTotal, t=step, g=self.environment.successful))

    def predict_game(self, agents):
        for agent in agents:
            agent.load_model_trained()
        
        state = self.environment.reset()
        state = np.array(state)
        state = state.ravel()

        done = False
        step = 0
        rewardTotal = 0

        while not done and step < self.max_timestep:
            action_dict = {}
            for agent in agents:
                legalActions = self.environment.legal_actions(agent.name)
                agent.legal_actions(legalActions)

                action = agent.predict(state)
                action_dict[agent.name] = action
            
            next_state, reward, done, info = self.environment.dqn_step(action_dict)
            next_state = np.array(next_state)
            next_state = next_state.ravel()

            state = next_state

            rewardTotal += reward
            step += 1
        
        print("Score:{s} with Steps:{t}, Goal:{g}".format(s=rewardTotal, t=step, g=self.environment.successful))
        
        return (self.environment.successful, rewardTotal, step)
            
    def set_alpha_and_epsilon(self, agents):
        for agent in agents:
            agent.set_alpha_and_epsilon()
        
    def final(self, agents):
        self.set_alpha_and_epsilon(agents)
        for i in range(self.final_episodes):
            self.run(agents)
    
    def set_filename(self, filename):
        file = './utils/dqn_result/{}'.format(filename)
        return file
    
    def filename_create_dlmodel(self):
        filename = "agent-{}-learningRate_{}-replay_{}-numTraining_{}-role_{}.h5".format(
            "dqn", 
            self.arglist.learning_rate, 
            self.arglist.replay,
            self.arglist.number_training,
            self.arglist.role,
        )
        return filename

    def filename_create_result(self):
        pass





    
    


