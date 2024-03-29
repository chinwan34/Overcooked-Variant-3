# from environment import OvercookedEnvironment
# from gym_cooking.envs import OvercookedEnvironment
from recipe_planner.recipe import *
from utils.world import World
from utils.agent import RealAgent, SimAgent, COLORS
from utils.core import *
from misc.game.gameplay import GamePlay
from misc.metrics.metrics_bag import Bag
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

    return parser.parse_args()


def fix_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def findSuitableRoles(actionsNotSatisfied, num_agents):
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
    
def roleAssignmentAlgorithm(typeUsed, num_agents):
    if typeUsed == "extreme":
        return [IdlePerson(), InvincibleWaiter()]
    elif typeUsed == "none":
        return [InvincibleWaiter(), InvincibleWaiter()]
    elif typeUsed == "unbalanced":
        return [CookingWaiter(), ExceptionalChefMerger()]
    elif typeUsed == "three":
        if num_agents == 2:
            return [ExceptionalChefMerger(), CookingMergingWaiter()]
        elif num_agents == 3:
            return [ChoppingWaiter(), ExceptionalChefMerger(), MergingWaiter()]


def initialize_agents(arglist):
    real_agents = []

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
                    roleList = roleAssignmentAlgorithm(arglist.role, arglist.num_agents)
                if (finished == False):
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
    return real_agents

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

if __name__ == '__main__':
    arglist = parse_arguments()
    if arglist.play:
        env = gym.envs.make("gym_cooking:overcookedEnv-v0", arglist=arglist)
        env.reset()
        game = GamePlay(env.filename, env.world, env.sim_agents)
        game.on_execute()
    else:
        model_types = [arglist.model1, arglist.model2, arglist.model3, arglist.model4]
        assert len(list(filter(lambda x: x is not None,
            model_types))) == arglist.num_agents, "num_agents should match the number of models specified"
        fix_seed(seed=arglist.seed)
        main_loop(arglist=arglist)


