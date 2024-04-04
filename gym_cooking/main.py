# from environment import OvercookedEnvironment
# from gym_cooking.envs import OvercookedEnvironment
from recipe_planner.recipe import *
from utils.world import World
from utils.agent import RealAgent, SimAgent, COLORS
from utils.core import *
from utils.DQN_main import mainAlgorithm
from misc.game.gameplay import GamePlay
from misc.metrics.metrics_bag import Bag
from utils.DQNagent import *
from random import randrange

import numpy as np
import random
import argparse
from collections import namedtuple

import gym


def parse_arguments():
    parser = argparse.ArgumentParser("Overcooked 2 argument parser")

    # Environment
    parser.add_argument("--level", type=str, required=True)
    parser.add_argument("--num-agents", type=int, required=True)
    parser.add_argument("--max-num-timesteps", type=int, default=100, help="Max number of timesteps to run")
    parser.add_argument("--max-num-subtasks", type=int, default=14, help="Max number of subtasks for recipe")
    parser.add_argument("--seed", type=int, default=1, help="Fix pseudorandom seed")
    parser.add_argument("--with-image-obs", action="store_true", default=False, help="Return observations as images (instead of objects)")
    parser.add_argument("--role", type=str, default=None, help="Role assignment for each play (optimal, unbalanced, extreme, three, none)")

    # Delegation Planner
    parser.add_argument("--beta", type=float, default=1.3, help="Beta for softmax in Bayesian delegation updates")

    # Navigation Planner
    parser.add_argument("--alpha", type=float, default=0.01, help="Alpha for BRTDP")
    parser.add_argument("--tau", type=int, default=2, help="Normalize v diff")
    parser.add_argument("--cap", type=int, default=75, help="Max number of steps in each main loop of BRTDP")
    parser.add_argument("--main-cap", type=int, default=100, help="Max number of main loops in each run of BRTDP")

    # Visualizations
    parser.add_argument("--play", action="store_true", default=False, help="Play interactive game with keys")
    parser.add_argument("--record", action="store_true", default=False, help="Save observation at each time step as an image in misc/game/record")

    # Models
    # Valid options: `bd` = Bayes Delegation; `up` = Uniform Priors
    # `dc` = Divide & Conquer; `fb` = Fixed Beliefs; `greedy` = Greedy
    parser.add_argument("--model1", type=str, default=None, help="Model type for agent 1 (bd, up, dc, fb, or greedy)")
    parser.add_argument("--model2", type=str, default=None, help="Model type for agent 2 (bd, up, dc, fb, or greedy)")
    parser.add_argument("--model3", type=str, default=None, help="Model type for agent 3 (bd, up, dc, fb, or greedy)")
    parser.add_argument("--model4", type=str, default=None, help="Model type for agent 4 (bd, up, dc, fb, or greedy)")

    # Arguments for DQN Learning
    parser.add_argument("--dqn", action="store_true", default=False, help="Use DQN to train the agents")
    parser.add_argument("--maxCapacity", type=int, default=100000, help="Maximum capacity of memory")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--update-frequency", type=int, default=10, help='The frequency of updates on target model')
    parser.add_argument("--replay", type=int, default=4, help="Steps difference for training")
    parser.add_argument("--number-training", default=10, type=int, help="Number of episodes for training")
    parser.add_argument("--learning-rate", default=0.00025, type=float, help="Learning rate of DQN")
    parser.add_argument("--game-play", default=2, type=int, help="Number of game play")
    parser.add_argument("--num-nodes", default=64, type=int, help="Number of nodes in each layer of DQN")


    return parser.parse_args()


def fix_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

# PROJECT INVOLVED THIS FUNCTION CHANGE.
def findSuitableRoles(actionsNotSatisfied, num_agents):
    """Allocate the optimal linear action sequence roles for the 
        environment. For project simulation, please only utilize 2 or 3 
        agents for functional results.
        Args:
            actionsNotSatisfied: List of actions that should be completed.
            num_agents: The number of agents in the simulation
        Return:
            The combination of roles to assign. 
    """
    listOfRoles = [Merger(), Chopper(), Deliverer(), Baker(), Cooker(), Cleaner(), Frier()]
    listOfRoles2 = [ChoppingWaiter(), MergingWaiter(), CookingWaiter(), ExceptionalChef(), BakingWaiter(), FryingWaiter()]
    SingleAgentRole = [InvincibleWaiter()]

    actionNamePair = [(Merge, "Merge"), (Get, "Get"), (Deliver, "Deliver"), (Cook, "Cook"), (Fry, "Fry"), (Chop, "Chop"),
                        (Bake, "Bake"), (Clean, "Clean")]

    if num_agents > 2:
            combinationsBasedOnAgents = combinations(listOfRoles, num_agents)
    elif num_agents == 2:
        combinationsBasedOnAgents = combinations(listOfRoles2, num_agents)
    elif num_agents == 1:
        combinationsBasedOnAgents = SingleAgentRole
        return SingleAgentRole

    for eachCombination in combinationsBasedOnAgents:
        currentSet = set()
        for i in eachCombination:
            initialized = i
            for action in initialized.probableActions:
                for (classType, stringUsed) in actionNamePair:
                    if action == classType:
                        currentSet.add(stringUsed)
        set.union(currentSet)

        if actionsNotSatisfied.issubset(currentSet):
            return eachCombination

# PROJECT INVOLVED THIS FUNCTION CHANGE.
def roleAssignmentAlgorithm(typeUsed, num_agents, level):
    """ The main role allocation system, allocate manually based
    on the level of time complexity required. 

    Note, for unbalanced / none / extreme, please only utilize two
    agents maximum in simulation; while for three, although it is
    possible for three agents, it is generally not possible for
    very-easy map due to limited space (Not implemented).

    Args:
        typeUsed: The role allocation mechanism
        num_agents: Number of agents in the environment
        level: The current level name
    
    Return:
        A list of role assignment for simulation

    """
    if typeUsed == "extreme":
        return [InvincibleWaiter(), IdlePerson()]
    
    elif typeUsed == "none":
        return [InvincibleWaiter(), InvincibleWaiter()]
    
    elif typeUsed == "unbalanced":
        if level.endswith("CF"):
            return [FryingWaiter(), ExceptionalChefMerger()]
        elif level.endswith("tomato") or level.endswith("salad"):
            return [Chopper(), InvincibleWaiter()]
        elif level.endswith("burger"):
            return [CookingWaiter(), ExceptionalChefMerger()]
        return [InvincibleWaiter(), InvincibleWaiter()]
    
    elif typeUsed == "three":
        if num_agents == 2:
            if level.endswith("CF"):
                return [ExceptionalChefMerger(), FryingMergingWaiter()]
            elif level.endswith("tomato") or level.endswith("salad"):
                return [ChoppingMerger(), MergingWaiter()]
            elif level.endswith("burger"):
                return [ExceptionalChefMerger(), CookingMergingWaiter()]
            return [InvincibleWaiter(), InvincibleWaiter()]
        
        elif num_agents == 3:
            if level.endswith("CF"):
                return [FryingWaiter(), ExceptionalChefMerger(), MergingWaiter()]
            elif level.endswith("tomato") or level.endswith("salad"):
                return [ChoppingWaiter(), Merger(), MergingWaiter()]
            elif level.endswith("burger"):
                return [CookingWaiter(), ExceptionalChefMerger(), CookingMergingWaiter()]
            return [InvincibleWaiter(), InvincibleWaiter(), InvincibleWaiter()]


def initialize_agents(arglist, state_size=0, action_size=0, dlmodel=None):
    real_agents = []
    dqn_agents = []

    with open('utils/levels/{}.txt'.format(arglist.level), 'r') as f:
        phase = 1
        recipes = []
        index = 0
        finished = False
        actionLeft = []

        for line in f:
            line = line.strip('\n')
            if line == '':
                phase += 1

            # phase 2: read in recipe list
            elif phase == 2:
                recipes.append(globals()[line]())

            # phase 3: read in agent locations (up to num_agents)
            elif phase == 3:
                if not actionLeft:
                    for i in recipes:
                        actionLeft = actionLeft + list(i.actions)
                    actionLeft = list(dict.fromkeys(actionLeft))
                    actionLeft = set(action.name for action in actionLeft)

                roleList = []
                if not arglist.role or arglist.role == "optimal":
                    roleList = findSuitableRoles(actionLeft, arglist.num_agents)
                else:
                    roleList = roleAssignmentAlgorithm(arglist.role, arglist.num_agents, arglist.level)
                if (finished == False and not arglist.dqn):
                    loc = line.split(' ')
                    real_agent = RealAgent(
                        arglist=arglist,
                        # name='agent-'+str(len(real_agents)+1)+roleList[index].name,
                        name='agent-'+str(len(real_agents)+1),
                        id_color=COLORS[len(real_agents)],
                        recipes=recipes,
                        role=roleList[index]
                    )
                    real_agents.append(real_agent)
                    if len(real_agents) >= arglist.num_agents:
                        finished = True
                    index+=1
                # DQN Agents specification
                elif (finished == False and arglist.dqn):
                    loc = line.split(' ')
                    dqn_agent = DQNAgent(
                        arglist=arglist,
                        st_size=state_size,
                        action_size=action_size,
                        name='agent-'+str(len(dqn_agents)+1),
                        color=COLORS[len(dqn_agents)],
                        role=roleList[index],
                        agent_index=len(dqn_agents),
                        dlmodel_name=dlmodel
                    )
                    dqn_agents.append(dqn_agent)
                    if len(dqn_agents) >= arglist.num_agents:
                        finished = True
                    index += 1
    if not arglist.dqn:
        return real_agents
    return dqn_agents

def main_loop(arglist):
    """The main loop for running experiments."""
    print("Initializing environment and agents.")
    env = gym.envs.make("gym_cooking:overcookedEnv-v0", arglist=arglist)
    obs = env.reset()
    # game = GameVisualize(env)
    real_agents = initialize_agents(arglist=arglist)
    # Info bag for saving pkl files
    bag = Bag(arglist=arglist, filename=env.filename)
    bag.set_recipe(recipe_subtasks=env.all_subtasks)

    while not env.done():
        action_dict = {}

        for agent in real_agents:
            action = agent.select_action(obs=obs)
            action_dict[agent.name] = action

        obs, reward, done, info = env.step(action_dict=action_dict)

        # Agents
        for agent in real_agents:
            agent.refresh_subtasks(world=env.world)

        # Saving info
        bag.add_status(cur_time=info['t'], real_agents=real_agents)


    # Saving final information before saving pkl file
    bag.set_collisions(collisions=env.collisions)
    bag.set_termination(termination_info=env.termination_info,
            successful=env.successful)

# PROJECT INVOLVED THIS FUNCTION CHANGE.
def dqn_main(arglist):
    """
    The main DQN simulation loop. Require
    argument specification for this to run.

    Args:
        arglist: The argument list in the command line

    """
    print("Initializing environment and agents.")
    env = gym.envs.make("gym_cooking:overcookedEnv-v0", arglist=arglist)
    obs = env.reset()

    dqnClass = mainAlgorithm(env, arglist)
    dqn_agents = []

    # Set file name at utils/dqn_result
    dlmodel_file = dqnClass.set_filename(dqnClass.filename_create_dlmodel())
    dlreward_file = dqnClass.set_filename_reward(dqnClass.filename_create_reward())
    dlstatistics_file = dqnClass.filename_create_statistics()
    state_size, action_size = env.world_size_action_size()
    dqn_agents = initialize_agents(arglist, state_size, action_size, dlmodel_file)

    # Main running algorithm
    # dqnClass.run(dqn_agents, dlreward_file)

    dones = []
    rewards = []
    time_steps = []

    # The testing run, if not wanting it, please set game_play = 0
    for i in range(arglist.game_play):
        (done, reward, step) = dqnClass.predict_game(dqn_agents)
        dones.append(done)
        rewards.append(reward)
        time_steps.append(step)
    
    if arglist.game_play > 0:
        print("Average score: ", sum(rewards)/len(rewards))
        print("Success Rate: ", dones.count(True)/len(dones))
        print("Average Time-step", sum(time_steps)/len(time_steps))

        dqnClass.store_statistics(dlstatistics_file, sum(time_steps)/len(time_steps), sum(rewards)/len(rewards), arglist)



if __name__ == '__main__':
    arglist = parse_arguments()
    if arglist.play:
        env = gym.envs.make("gym_cooking:overcookedEnv-v0", arglist=arglist)
        env.reset()
        game = GamePlay(env.filename, env.world, env.sim_agents)
        game.on_execute()
    elif arglist.dqn:
        dqn_main(arglist)
    else:
        model_types = [arglist.model1, arglist.model2, arglist.model3, arglist.model4]
        assert len(list(filter(lambda x: x is not None,
            model_types))) == arglist.num_agents, "num_agents should match the number of models specified"
        fix_seed(seed=arglist.seed)
        main_loop(arglist=arglist)


