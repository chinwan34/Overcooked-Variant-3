import numpy as np
import random
from utils.DQNagent import DQNAgent
import pandas as pd

class mainAlgorithm:
    """
    The main algorithm for DQN simulation.
    """
    def __init__(self, environment, arglist):
        self.arglist = arglist
        self.environment = environment
        self.num_training = self.arglist.number_training
        self.max_timestep = self.arglist.max_num_timesteps
        self.filling_step = 0
        self.update_step = self.arglist.replay
        self.final_episodes = 5

    def run(self, agents, dlreward_file):
        """
        The simulation procedure, with arguments
        specified. After all simulations finished, the relevant 
        statistics are updated and printed.

        Args:
            agents: The DQNAgents for simulation and update
            dlreward_file: Filename for storing rewards on this run
        """
        time_steps = []
        all_step = 0
        rewards = []
        maxScore = float("-inf")
        for episode in range(self.num_training):
            print("EPISODE------------", episode, "-----------EPISODE")
            state = self.environment.reset()
        
            done = False
            rewardTotal = 0
            step = 0
            state = np.array(state)
            state = state.ravel()

            while not done and step < self.max_timestep:
                action_dict = {}
                for agent in agents:
                    # Legal actions check
                    legalActions = self.environment.legal_actions(agent.name)
                    agent.legal_actions(legalActions)

                    action = agent.epsilon_greedy(state)
                    action_dict[agent.name] = action
                
                # Main step
                next_state, reward, doneUsed, info = self.environment.dqn_step(action_dict)
                next_state = np.array(next_state)
                next_state = next_state.ravel()

                # Observe and update memory
                for agent in agents:
                    agent.observeTransition((state, action_dict, reward, next_state, doneUsed))
                    agent.epsilon_decay()
                    # only update at a certain update frequency
                    if step % self.update_step == 0:
                        agent.update_result()
                    agent.update_target()
                
                rewardTotal += reward
                done = doneUsed
                all_step += 1
                step += 1
                state = next_state
            
            time_steps.append(step)
            rewards.append(rewardTotal)

            if episode % 100 == 0:
                # maximum score update
                if rewardTotal > maxScore:
                    for agent in agents:
                        print("Got in episode for updates")
                        agent.dlmodel.save_model()               
                    maxScore = rewardTotal
            
            if episode % 100 == 0:
                df = pd.DataFrame(rewards, columns=['currScore'])
                df.to_csv(dlreward_file)
            
            print("Final score:{}, Steps:{}, and whether goal reached:{}".format(rewardTotal, step, self.environment.successful))
            

    def predict_game(self, agents):
        """
        Setting alpha, epsilon to 0 for no exploration,
        test model on the original environment. It is a deterministic
        environment for training.

        Args:
            agents: The DQNAgents for simulation
        Return:
            List of whether the result is successful, total reward, steps taken
        """
        for agent in agents:
            agent.load_model_trained()
        
        state = self.environment.reset()
        state = np.array(state)
        state = state.ravel()

        done = False
        rewardTotal = 0
        step = 0

        while not done and step < self.max_timestep:
            action_dict = {}
            for agent in agents:
                # Only allow legal actions
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
        
        return (self.environment.successful, rewardTotal, step)
            
    def set_alpha_and_epsilon(self, agents):
        """
        Set the alpha and epsilon to 0 for testing,
        currently archived.

        Args:
            agents: The DQNAgents for simulation and update
        """
        for agent in agents:
            agent.set_alpha_and_epsilon()
    
    def set_filename(self, filename):
        """
        Set the filename for the location of model stored.

        Args:
            filename: The name specified for storage
        Return:
            The file name
        """
        file1 = './utils/dqn_result/{}'.format(filename[0])
        file2 = './utils/dqn_result/{}'.format(filename[1])
        return [file1, file2]
    
    def set_filename_reward(self, filename):
        """
        Set the filename for the rewards accumulated.

        Args:
            filename: The name specified for storage
        Return:
            The file name
        """
        file = './utils/dqn_reward/{}'.format(filename)
        return file
    
    def filename_create_dlmodel(self):
        """
        Create the filename of the model.
        Return:
            Filename created
        """
        filename1 = "agent-{}-learningRate_{}-replay_{}-numTraining_{}-role_{}-agent_1.h5".format(
            "dqn", 
            self.arglist.learning_rate, 
            self.arglist.replay,
            self.arglist.number_training,
            self.arglist.role
        )
        filename2 = "agent-{}-learningRate_{}-replay_{}-numTraining_{}-role_{}-agent_2.h5".format(
            "dqn", 
            self.arglist.learning_rate, 
            self.arglist.replay,
            self.arglist.number_training,
            self.arglist.role
        )
        return [filename1, filename2]

    def filename_create_reward(self):
        """
        Create the reward file of the model.
        Return:
            Filename created
        """
        filename = "reward-agent-{}-learningRate_{}-replay_{}-numTraining_{}-role_{}.csv".format(
            "dqn", 
            self.arglist.learning_rate, 
            self.arglist.replay,
            self.arglist.number_training,
            self.arglist.role,
        )
        return filename

    def filename_create_statistics(self):
        """
        Create the statistics file of the model.
        Return:
            Filename created
        """
        filename = "./utils/dqn_reward/statistics_file.csv"
        return filename
    
    def store_statistics(self, filename, steps, rewards, arglist):
        """
        Store the statistics after simulation into a csv file.
        
        Args:
            filename: The file name for statistic storage
            steps: Average steps for simulation result
            rewards: Average rewards for simulation result
            arglist: List of arguments specified
        """
        try:
            df = pd.read_csv(filename)
        except FileNotFoundError:
            df = pd.DataFrame(columns=["Episodes", "Test number", "Alpha", "Level", "Role", "Steps", "Rewards"])
        df = df.append({
            "Episodes": arglist.number_training,
            "Test number": arglist.game_play,
            "Alpha": arglist.learning_rate,
            "Level": arglist.level,
            "Role": arglist.role,
            "Steps": steps,
            "Rewards": rewards
        }, ignore_index=True)

        df.to_csv(filename)

    def filename_create_graphs(self, filename):
        """
        Create the graphs file of the model.
        Return:
            Filename created
        """
        filename = "./utils/dqn_reward/statistics_file.csv"
        return filename

