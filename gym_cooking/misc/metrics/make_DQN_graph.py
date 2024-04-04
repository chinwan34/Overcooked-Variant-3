import argparse
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import sys

def parse_arguments():
    parser = argparse.ArgumentParser(description="for parsing")
    parser.add_argument("--reward-episodes", action="store_true", default="False", help="Reward-episodes graph")
    parser.add_argument("--reward-legend", action="store_true", default="False", help="Average reward")
    parser.add_argument("--timesteps-legend", action="store_true", default="False", help="Average TimeSteps")
    parser.add_argument("--reward-alpha", action="store_true", default="False", help="Reward to alpha")
    parser.add_argument("--lr", type=float, default=0.00025, help="Only for reward legend, learning rate of simulation")
    parser.add_argument("--replay", type=int, default=4, help="Only for reward legend, replay step")
    parser.add_argument("--numTraining", type=int, default=10, help="Only for reward legend, number of episodes")
    parser.add_argument("--role", type=str, default="none", help="Only for reward legend, role for simulation")
    return parser.parse_args()

def main_loop():
    """
    Plot and store the relevant graphs based on argument list.
    """
    path_save = os.path.join(os.getcwd(), 'dqn_graphs')
    if not os.path.exists(path_save):
        os.makedirs(path_save)
    
    if arglist.reward_legend:
        df = import_data(arglist, reward=True)
    else:
        df = import_data(arglist, reward=False)
    plot_graph(df, arglist, path_save)

def import_data(arglist, reward=False):
    """
    Import the relevant data from folder for simulations.

    Args:
        arglist: Argument list for functioning
        reward: Whether to impose data retrieval from a different file
    Return:
        The dataframe for graph creation
    """
    if reward:
        try:
            df = pd.read_csv("../../utils/dqn_reward/reward-agent-dqn-learningRate_{}-replay_{}-numTraining_{}-role_{}.csv".format(
                arglist.lr,
                arglist.replay,
                arglist.numTraining,
                arglist.role
            ))
            return df
        except FileNotFoundError:
            print("Error loading reward results file")
            sys.exit(1)
    else:
        try:
            df = pd.read_csv("../../utils/dqn_reward/statistics_file.csv")
            return df
        except FileNotFoundError:
            print("Error loading statistics file.")
            sys.exit(1)
    

def plot_graph(df, arglist, path_save):
    color_palette = sns.color_palette()
    sns.set_style('ticks')
    sns.set_context('talk', font_scale=1)

    if arglist.reward_legend:
        plt.figure(figsize=(10, 10))
        try:
            score = df['currScore']
            plt.plot(range(len(score)), score)
            plt.xlabel("Episodes")
            plt.ylabel("reward")
            plt.savefig(os.path.join(path_save, 'reward-legend-lr_{}_replay_{}-numTraining_{}-role_{}.png'.format(
                arglist.lr,
                arglist.replay,
                arglist.numTraining,
                arglist.role,
            )))
            print("Completed reward legend graph storage.")
            plt.close()
        except KeyError:
            print("No current score yet. Please simulate first.")
            sys.exit(1)

        return

    if arglist.reward_episodes:
        plt.figure(figsize=(10,10))
        try:
            steps = df['Steps']
            episodes = df['Episodes']
        except KeyError:
            print("No steps or episodes column found.")
            sys.exit(1)

        plt.plot(episodes, steps)
        plt.scatter(episodes, steps)
        plt.xlabel("episodes")
        plt.ylabel("steps")
        plt.savefig(os.path.join(path_save, 'steps-episodes-legend.png'), dpi="figure")
        
        print("Completed reward episodes graph storage.")
        plt.close()

    if arglist.timesteps_legend:
        pass
    if arglist.reward_alpha:
        pass


if __name__ == "__main__":
    arglist = parse_arguments()
    main_loop()

    


