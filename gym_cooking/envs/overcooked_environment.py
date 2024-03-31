# Recipe planning
from recipe_planner.stripsworld import STRIPSWorld
import recipe_planner.utils as recipe
from recipe_planner.recipe import *

# Delegation planning
from delegation_planner.bayesian_delegator import BayesianDelegator

# Navigation planning
import navigation_planner.utils as nav_utils

# Other core modules
from utils.interact import interact
from utils.world import World
from utils.core import *
from utils.agent import SimAgent
from misc.game.gameimage import GameImage
from utils.agent import COLORS

import copy
import networkx as nx
import numpy as np
from itertools import combinations, permutations, product
from collections import namedtuple

import gym
from gym import error, spaces, utils
from gym.utils import seeding


CollisionRepr = namedtuple("CollisionRepr", "time agent_names agent_locations")


class OvercookedEnvironment(gym.Env):
    """Environment object for Overcooked."""

    def __init__(self, arglist):
        self.arglist = arglist
        self.t = 0
        self.set_filename()

        self.widthOfGame = 0
        self.heightOfGame = 0

        

        # For visualizing episode.
        self.rep = []
        self.repDQN = []

        # For tracking data during an episode.
        self.collisions = []
        self.termination_info = ""
        self.successful = False

    def get_repr(self):
        return self.world.get_repr() + tuple([agent.get_repr() for agent in self.sim_agents])

    def __str__(self):
        # Print the world and agents.
        _display = list(map(lambda x: ''.join(map(lambda y: y + ' ', x)), self.rep))
        return '\n'.join(_display)

    def __eq__(self, other):
        # return self.repDQN_conv == other.repDQN_conv
        return self.get_repr() == other.get_repr() or self.repDQN_conv == other.repDQN_conv
    
    def __hash__(self):
        return self.repDQN_conv.__hash__()

    def __copy__(self):
        new_env = OvercookedEnvironment(self.arglist)
        new_env.__dict__ = self.__dict__.copy()
        new_env.world = copy.copy(self.world)
        new_env.sim_agents = [copy.copy(a) for a in self.sim_agents]
        new_env.distances = self.distances

        # Make sure new objects and new agents' holdings have the right pointers.
        for a in new_env.sim_agents:
            if a.holding is not None:
                a.holding = new_env.world.get_object_at(
                        location=a.location,
                        desired_obj=None,
                        find_held_objects=True)
        return new_env

    def set_filename(self):
        self.filename = "{}_agents{}_seed{}".format(self.arglist.level,\
            self.arglist.num_agents, self.arglist.seed)
        model = ""
        if self.arglist.model1 is not None:
            model += "_model1-{}".format(self.arglist.model1)
        if self.arglist.model2 is not None:
            model += "_model2-{}".format(self.arglist.model2)
        if self.arglist.model3 is not None:
            model += "_model3-{}".format(self.arglist.model3)
        if self.arglist.model4 is not None:
            model += "_model4-{}".format(self.arglist.model4)
        self.filename += model

        if self.arglist.role is not None:
            self.filename += "role-{}".format(self.arglist.role)

    def load_level(self, level, num_agents):
        x = 0
        y = 0
        actionsNotSatisfied = []
        with open('utils/levels/{}.txt'.format(level), 'r') as file:
            # Mark the phases of reading.
            phase = 1
            AgentsDone = False
            currentIndex=0
            for line in file:
                line = line.strip('\n')
                if line == '':
                    phase += 1

                # Phase 1: Read in kitchen map.
                elif phase == 1:
                    for x, rep in enumerate(line):
                        # Object, i.e. Tomato, Lettuce, Onion, or Plate.
                        if rep in 'tlopkfmbcP':
                            counter = Counter(location=(x, y))

                            obj = Object(
                                    location=(x, y),
                                    contents=RepToClass[rep]())
                            counter.acquire(obj=obj)
                            self.world.insert(obj=counter)
                            self.world.insert(obj=obj)
                        # GridSquare, i.e. Floor, Counter, Cutboard, Delivery.
                        elif rep in RepToClass:
                            newobj = RepToClass[rep]((x, y))
                            self.world.objects.setdefault(newobj.name, []).append(newobj)
                        else:
                            # Empty. Set a Floor tile.
                            f = Floor(location=(x, y))
                            self.world.objects.setdefault('Floor', []).append(f)
                    y += 1
                # Phase 2: Read in recipe list.
                elif phase == 2:
                    self.recipes.append(globals()[line]())

                # Phase 3: Read in agent locations (up to num_agents).
                elif phase == 3:
                    if not actionsNotSatisfied:
                        for i in self.recipes:
                            actionsNotSatisfied = actionsNotSatisfied + list(i.actions)
                        
                        actionsNotSatisfied = list(dict.fromkeys(actionsNotSatisfied))
                        actionsNotSatisfied = set(action.name for action in actionsNotSatisfied)
                    
                    roleList = []
                    if not self.arglist.role or self.arglist.role == "optimal":
                        roleList = self.findSuitableRoles(actionsNotSatisfied, num_agents)
                    else:
                        roleList = self.roleAssignmentAlgorithm(self.arglist.role, num_agents)
                    if (AgentsDone == False):
                        loc = line.split(' ')
                        sim_agent = SimAgent(
                            # name='agent-'+str(len(self.sim_agents)+1)+roleList[currentIndex].name,
                            name='agent-'+str(len(self.sim_agents)+1),
                            role=roleList[currentIndex],
                            id_color=COLORS[len(self.sim_agents)],
                            location=(int(loc[0]), int(loc[1])))
                        self.sim_agents.append(sim_agent)
                        currentIndex+=1
                        if (len(self.sim_agents)) >= num_agents:
                            AgentsDone = True

        self.distances = {}
        self.world.width = x+1
        self.world.height = y
        self.world.perimeter = 2*(self.world.width + self.world.height)

        self.widthOfGame = x+1
        self.heightOfGame = y

    def findSuitableRoles(self, actionsNotSatisfied, num_agents):
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
    
    def roleAssignmentAlgorithm(self, typeUsed, num_agents):
        if typeUsed == "extreme":
            return [InvincibleWaiter(), IdlePerson()]
        elif typeUsed == "none":
            return [InvincibleWaiter(), InvincibleWaiter()]
        elif typeUsed == "unbalanced":
            return [CookingWaiter(), ExceptionalChefMerger()]
        elif typeUsed == "three":
            if num_agents == 2:
                return [ExceptionalChefMerger(), CookingMergingWaiter()]
            elif num_agents == 3:
                return [ChoppingWaiter(), ExceptionalChefMerger(), MergingWaiter()]

    def reset(self):
        self.world = World(arglist=self.arglist)
        self.recipes = []
        self.sim_agents = []
        self.agent_actions = {}
        self.t = 0

        # For visualizing episode.
        self.rep = []
        self.repDQN = []
        self.repDQN_conv = np.zeros((self.widthOfGame, self.heightOfGame, 4))

        # For tracking data during an episode.
        self.collisions = []
        self.termination_info = ""
        self.successful = False

        print("Before load level")

        # Load world & distances.
        self.load_level(
                level=self.arglist.level,
                num_agents=self.arglist.num_agents)
        print("On env.reset location")
        self.all_subtasks = self.run_recipes()
        self.subtasks_left = self.all_subtasks
        self.world.make_loc_to_gridsquare()
        self.world.make_reachability_graph()
        self.cache_distances()
        self.obs_tm1 = copy.copy(self)

        if self.arglist.record or self.arglist.with_image_obs:
            self.game = GameImage(
                    filename=self.filename,
                    world=self.world,
                    sim_agents=self.sim_agents,
                    record=self.arglist.record)
            self.game.on_init()
            if self.arglist.record:
                self.game.save_image_obs(self.t)
        if not self.arglist.dqn:
            return copy.copy(self)
        else:
            self.update_display_DQN_conv()   
            self.update_display_DQN()
            # return self.repDQN
            return self.repDQN_conv

    def world_size_action_size(self):
        # return len(self.repDQN), 4
        return len(self.repDQN_conv.flatten()), 4

    def close(self):
        return

    def step(self, action_dict):
        # Track internal environment info.
        self.t += 1
        print("===============================")
        print("[environment.step] @ TIMESTEP {}".format(self.t))
        print("===============================")

        # Get actions.
        for sim_agent in self.sim_agents:
            sim_agent.action = action_dict[sim_agent.name]

        # Check collisions.
        self.check_collisions()
        self.obs_tm1 = copy.copy(self)

        # Execute.
        self.execute_navigation()

        # Visualize.
        self.display()
        self.print_agents()
        if self.arglist.record:
            self.game.save_image_obs(self.t)

        # Get a plan-representation observation.
        new_obs = copy.copy(self)
        # Get an image observation
        image_obs = self.game.get_image_obs()

        done = self.done()
        reward = self.reward()
        info = {"t": self.t, "obs": new_obs,
                "image_obs": image_obs,
                "done": done, "termination_info": self.termination_info}
        return new_obs, reward, done, info

    def dqn_step(self, action_dict):
        self.t += 1
        print("===============================")
        print("[environment.step] @ TIMESTEP {}".format(self.t))
        print("===============================")

        # Choose actions, may not to fix sim_agents representation
        print(action_dict)

        actions_available = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for sim_agent in self.sim_agents:
            if not isinstance(action_dict[sim_agent.name], tuple):
                sim_agent.action = actions_available[action_dict[sim_agent.name]]
            else:
                sim_agent.action = action_dict[sim_agent.name]
        
        # Execute.
        self.execute_navigation()

        # May need to Visualize and store model
        # new_obs = copy.copy(self)

        # States, rewards, done
        done = self.done()
        reward = self.dqn_reward()
        self.subtask_reduction()
        # self.update_display_DQN()
        self.update_display_DQN_conv()
        self.update_display()

        if self.arglist.record:
            self.game.save_image_obs(self.t)
        # next_state = self.repDQN
        next_state = self.repDQN_conv

        info = {
            "t": self.t,
            # "obs": new_obs,
            "done": done, "termination_info": self.termination_info
        }

        return next_state, reward, done, info

    def subtask_reduction(self):
        delete = []
        for subtask in self.subtasks_left:
            _, goal_obj = nav_utils.get_subtask_obj(subtask)
            goal_obj_locs = self.world.get_all_object_locs(obj=goal_obj)
            delivery_loc = list(filter(lambda o: o.name=='Delivery', self.world.get_object_list()))[0].location
            if isinstance(subtask, recipe.Deliver):
                if any([gol == delivery_loc for gol in goal_obj_locs]):
                    delete.append(subtask)
            elif len(goal_obj_locs) != 0:
                delete.append(subtask)
        
        if len(delete) > 0:
            self.subtasks_left.remove(delete[0])     
            return True
        return False  
    
    def single_subtask_reduction(self, subtask):
        doneCheck = False
        _, goal_obj = nav_utils.get_subtask_obj(subtask)
        goal_obj_locs = self.world.get_all_object_locs(obj=goal_obj)
        delivery_loc = list(filter(lambda o: o.name=='Delivery', self.world.get_object_list()))[0].location
        if isinstance(subtask, recipe.Deliver):
             if any([gol == delivery_loc for gol in goal_obj_locs]):
                 doneCheck = True
                 return True, doneCheck
        elif len(goal_obj_locs) != 0:
            return True, doneCheck
        return False, doneCheck

    def holding_important_object(self, subtask_agent_names, subtask):
        bonus = 0
        start_obj, goal_obj = nav_utils.get_subtask_obj(subtask)
        for agent in self.sim_agents:
            if agent.name in subtask_agent_names:
                if agent.holding == start_obj:
                    bonus += 1
                elif agent.holding == goal_obj:
                    bonus += 5
        return bonus
    
    def role_bonus(self, subtask_agent_names, subtask):
        bonus = 0
        start_obj, goal_obj = nav_utils.get_subtask_obj(subtask)
        for agent in self.sim_agents:
            if agent.name in subtask_agent_names:
                if any([isinstance(subtask, action) for action in agent.role.probableActions]):
                    if agent.holding == start_obj: bonus += 2
                    if agent.holding == goal_obj: bonus += 5
                else:
                    if agent.holding == start_obj: bonus -= 2
                    if agent.holding == goal_obj: bonus -= 5
        
        return bonus


    def done(self):
        # Done if the episode maxes out
        if self.t >= self.arglist.max_num_timesteps and self.arglist.max_num_timesteps:
            self.termination_info = "Terminating because passed {} timesteps".format(
                    self.arglist.max_num_timesteps)
            self.successful = False
            return True

        assert any([isinstance(subtask, recipe.Deliver) for subtask in self.all_subtasks]), "no delivery subtask"

        # Done if subtask is completed.
        for subtask in self.all_subtasks:
            # Double check all goal_objs are at Delivery.
            if isinstance(subtask, recipe.Deliver):
                _, goal_obj = nav_utils.get_subtask_obj(subtask)

                delivery_loc = list(filter(lambda o: o.name=='Delivery', self.world.get_object_list()))[0].location
                goal_obj_locs = self.world.get_all_object_locs(obj=goal_obj)
                if not any([gol == delivery_loc for gol in goal_obj_locs]):
                    self.termination_info = ""
                    self.successful = False
                    return False

        self.termination_info = "Terminating because all deliveries were completed"
        self.successful = True
        return True

    def reward(self):
        return 1 if self.successful else 0

    def dqn_reward(self):
        """
        I wrote this.
        """
        reward = 0
        for subtask in self.subtasks_left:
            finishedSubtask, doneCheck = self.single_subtask_reduction(subtask)
            if finishedSubtask:
                reward += 10
                if doneCheck: reward += 20
            else:
                reward -= 1
            start_obj, goal_obj = nav_utils.get_subtask_obj(subtask)
            subtask_action_obj = nav_utils.get_subtask_action_obj(subtask)
            distance = self.get_lower_bound_for_subtask_given_objs(
                subtask, 
                ["agent-1", "agent-2"],
                start_obj,
                goal_obj,
                subtask_action_obj,
            )
            reward -= 0.3 * distance

            # bonus = self.holding_important_object(["agent-1", "agent-2"], subtask)
            bonus2 = self.role_bonus(["agent-1", "agent-2"], subtask)
            reward = reward + bonus2
        
        return reward

    def print_agents(self):
        for sim_agent in self.sim_agents:
            sim_agent.print_status()

    def display(self):
        self.update_display()
        print(str(self))

    def update_display(self):
        self.rep = self.world.update_display()
        for agent in self.sim_agents:
            x, y = agent.location
            self.rep[y][x] = str(agent)

    def update_display_DQN(self):
        self.repDQN = self.world.update_display_dqn()
        for agent in self.sim_agents:
            x, y = agent.location
            # self.repDQN[x,y,3] = 1
            self.repDQN.append(y)
            self.repDQN.append(x)
    
    def update_display_DQN_conv(self):
        self.repDQN_conv = self.world.update_display_dqn_conv(self.widthOfGame, self.heightOfGame)
        for agent in self.sim_agents:
            x, y = agent.location
            self.repDQN_conv[x,y,3] = 1

    def get_agent_names(self):
        return [agent.name for agent in self.sim_agents]
    
    def get_agent_role_names(self):
        return [(agent.name, agent.role) for agent in self.sim_agents]

    def run_recipes(self):
        """Returns different permutations of completing recipes."""
        self.sw = STRIPSWorld(world=self.world, recipes=self.recipes)
        # [path for recipe 1, path for recipe 2, ...] where each path is a list of actions
        subtasks = self.sw.get_subtasks(max_path_length=self.arglist.max_num_subtasks)
        all_subtasks = [subtask for path in subtasks for subtask in path]
        print('Subtasks:', all_subtasks, '\n')
        return all_subtasks

    def get_AB_locs_given_objs(self, subtask, subtask_agent_names, start_obj, goal_obj, subtask_action_obj):
        """Returns list of locations relevant for subtask's Merge operator.

        See Merge operator formalism in our paper, under Fig. 11:
        https://arxiv.org/pdf/2003.11778.pdf"""

        # For Merge operator on Chop subtasks, we look at objects that can be
        # chopped and the cutting board objects.
        if isinstance(subtask, recipe.Chop) or \
            isinstance(subtask, recipe.Fry) or isinstance(subtask, recipe.Cook) or isinstance(subtask, recipe.Bake):
            # A: Object that can be chopped.
            A_locs = self.world.get_object_locs(obj=start_obj, is_held=False) + list(map(lambda a: a.location,\
                list(filter(lambda a: a.name in subtask_agent_names and a.holding == start_obj, self.sim_agents))))
            # print(A_locs)

            # B: Cutboard objects.
            B_locs = self.world.get_all_object_locs(obj=subtask_action_obj)

        elif isinstance(subtask, recipe.Clean):
            A_locs = self.world.get_object_locs_plate(obj=start_obj, is_held=False) + list(map(lambda a: a.location,\
                list(filter(lambda a: a.name in subtask_agent_names and a.holding == start_obj 
                and len(start_obj.contents) == 1
                and isinstance(start_obj.contents[0], Plate)
                and start_obj.contents[0].state_index == 0, self.sim_agents))))

            # B: Cutboard objects.
            B_locs = self.world.get_all_object_locs(obj=subtask_action_obj)
            
            # for diffobject in self.world.get_object_list():
            #     if isinstance(diffobject, Object) and len(diffobject.contents) == 1 and isinstance(diffobject.contents[0], Plate):
            #         print(diffobject.contents[0].state_index)
                    
                # if isinstance(diffobject, Plate):
                #     print("Plate state index", diffobject.state_index)
            
        # For Merge operator on Deliver subtasks, we look at objects that can be
        # delivered and the Delivery object.
        elif isinstance(subtask, recipe.Deliver):
            # B: Delivery objects.
            B_locs = self.world.get_all_object_locs(obj=subtask_action_obj)

            # A: Object that can be delivered.
            A_locs = self.world.get_object_locs(
                    obj=start_obj, is_held=False) + list(
                            map(lambda a: a.location, list(
                                filter(lambda a: a.name in subtask_agent_names and a.holding == start_obj, self.sim_agents))))
            A_locs = list(filter(lambda a: a not in B_locs, A_locs))

        # For Merge operator on Merge subtasks, we look at objects that can be
        # combined together. These objects are all ingredient objects (e.g. Tomato, Lettuce).
        elif isinstance(subtask, recipe.Merge):
            A_locs = self.world.get_object_locs(
                    obj=start_obj[0], is_held=False) + list(
                            map(lambda a: a.location, list(
                                filter(lambda a: a.name in subtask_agent_names and a.holding == start_obj[0], self.sim_agents))))
            B_locs = self.world.get_object_locs(
                    obj=start_obj[1], is_held=False) + list(
                            map(lambda a: a.location, list(
                                filter(lambda a: a.name in subtask_agent_names and a.holding == start_obj[1], self.sim_agents))))

        else:
            return [], []

        return A_locs, B_locs

    def get_lower_bound_for_subtask_given_objs(
            self, subtask, subtask_agent_names, start_obj, goal_obj, subtask_action_obj):
        """Return the lower bound distance (shortest path) under this subtask between objects."""
        assert len(subtask_agent_names) <= 2, 'passed in {} agents but can only do 1 or 2'.format(len(agents))

        # Calculate extra holding penalty if the object is irrelevant.
        holding_penalty = 0.0
        for agent in self.sim_agents:
            if agent.name in subtask_agent_names:
                # Check for whether the agent is holding something.
                if agent.holding is not None:
                    if isinstance(subtask, recipe.Merge):
                        continue
                    else:
                        if agent.holding != start_obj and agent.holding != goal_obj:
                            # Add one "distance"-unit cost
                            holding_penalty += 1.0
        # Account for two-agents where we DON'T want to overpenalize.
        holding_penalty = min(holding_penalty, 1)

        # Get current agent locations.
        agent_locs = [agent.location for agent in list(filter(lambda a: a.name in subtask_agent_names, self.sim_agents))]
        A_locs, B_locs = self.get_AB_locs_given_objs(
                subtask=subtask,
                subtask_agent_names=subtask_agent_names,
                start_obj=start_obj,
                goal_obj=goal_obj,
                subtask_action_obj=subtask_action_obj)

        # Add together distance and holding_penalty.
        return self.world.get_lower_bound_between(
                subtask=subtask,
                agent_locs=tuple(agent_locs),
                A_locs=tuple(A_locs),
                B_locs=tuple(B_locs)) + holding_penalty

    def nextLocationBase(self, agent_action, currentLocation):
        return self.world.get_gridsquare_at(location=tuple(np.asarray(currentLocation) + np.asarray(agent_action)))
    
    def legal_actions(self, agent_name):
        actions = [(0,1), (0,-1), (1,0), (-1,0)]
        legal_actions = []
        for agent in self.sim_agents:
            if agent.name == agent_name:
                if agent.holding:
                    return actions
                else:
                    for action in actions:
                        if self.world.is_object_at_location(location=tuple(np.asarray(agent.location) + np.asarray(action))):
                            legal_actions.append(action)
                        if isinstance(self.nextLocationBase(action, agent.location), Floor):
                            if action not in legal_actions:
                                legal_actions.append(action)
        
        if not legal_actions: legal_actions.append(random.choice(actions))
        return legal_actions
                    


    def is_occupied_location(self, agent_action, currentLocation):
        return self.world.is_occupied(location=tuple(np.asarray(currentLocation) + np.asarray(agent_action))) or self.world.is_delivery(location=tuple(np.asarray(currentLocation) + np.asarray(agent_action)))

    def is_collision(self, agent1_loc, agent2_loc, agent1_action, agent2_action):
        """Returns whether agents are colliding.

        Collisions happens if agent collide amongst themselves or with world objects."""
        # Tracks whether agents can execute their action.
        execute = [True, True]

        # Collision between agents and world objects.
        agent1_next_loc = tuple(np.asarray(agent1_loc) + np.asarray(agent1_action))
        if self.world.get_gridsquare_at(location=agent1_next_loc).collidable:
            # Revert back because agent collided.
            agent1_next_loc = agent1_loc

        agent2_next_loc = tuple(np.asarray(agent2_loc) + np.asarray(agent2_action))
        if self.world.get_gridsquare_at(location=agent2_next_loc).collidable:
            # Revert back because agent collided.
            agent2_next_loc = agent2_loc

        # Inter-agent collision.
        if agent1_next_loc == agent2_next_loc:
            if agent1_next_loc == agent1_loc and agent1_action != (0, 0):
                execute[1] = False
            elif agent2_next_loc == agent2_loc and agent2_action != (0, 0):
                execute[0] = False
            else:
                execute[0] = False
                execute[1] = False

        # Prevent agents from swapping places.
        elif ((agent1_loc == agent2_next_loc) and
                (agent2_loc == agent1_next_loc)):
            execute[0] = False
            execute[1] = False
        return execute
    
    def is_collision_alter(self, agent1_loc, agent2_loc, agent1_action, agent2_action):
        execute = [True, True]
        agent1_next_loc = tuple(np.asarray(agent1_loc) + np.asarray(agent1_action))
        if self.world.get_gridsquare_at(location=agent1_next_loc).collidable:
            # Revert back because agent collided.
            agent1_next_loc = agent1_loc

        agent2_next_loc = tuple(np.asarray(agent2_loc) + np.asarray(agent2_action))
        if self.world.get_gridsquare_at(location=agent2_next_loc).collidable:
            # Revert back because agent collided.
            # print(type(self.world.get_gridsquare_at(location=agent2_next_loc)))
            agent2_next_loc = agent2_loc
        
        return execute

    def check_collisions(self):
        """Checks for collisions and corrects agents' executable actions.

        Collisions can either happen amongst agents or between agents and world objects."""
        execute = [True for _ in self.sim_agents]

        # Check each pairwise collision between agents.
        for i, j in combinations(range(len(self.sim_agents)), 2):
            agent_i, agent_j = self.sim_agents[i], self.sim_agents[j]
            exec_ = self.is_collision(
                    agent1_loc=agent_i.location,
                    agent2_loc=agent_j.location,
                    agent1_action=agent_i.action,
                    agent2_action=agent_j.action)

            # Update exec array and set path to do nothing.
            if not exec_[0]:
                execute[i] = False
            if not exec_[1]:
                execute[j] = False

            # Track collisions.
            if not all(exec_):
                collision = CollisionRepr(
                        time=self.t,
                        agent_names=[agent_i.name, agent_j.name],
                        agent_locations=[agent_i.location, agent_j.location])
                self.collisions.append(collision)

        print('\nexecute array is:', execute)

        # Update agents' actions if collision was detected.
        for i, agent in enumerate(self.sim_agents):
            if not execute[i]:
                agent.action = (0, 0)
            print("{} has action {}".format(color(agent.name, agent.color), agent.action))

    def execute_navigation(self):
        for agent in self.sim_agents:
            interact(agent=agent, world=self.world)
            self.agent_actions[agent.name] = agent.action


    def cache_distances(self):
        """Saving distances between world objects."""
        counter_grid_names = [name for name in self.world.objects if "Supply" in name or "Counter" in name or "Delivery" in name or "Cut" in name]
        # Getting all source objects.
        source_objs = copy.copy(self.world.objects["Floor"])
        for name in counter_grid_names:
            source_objs += copy.copy(self.world.objects[name])
        # Getting all destination objects.
        dest_objs = source_objs

        # From every source (Counter and Floor objects),
        # calculate distance to other nodes.
        for source in source_objs:
            self.distances[source.location] = {}
            # Source to source distance is 0.
            self.distances[source.location][source.location] = 0
            for destination in dest_objs:
                # Possible edges to approach source and destination.
                source_edges = [(0, 0)] if not source.collidable else World.NAV_ACTIONS
                destination_edges = [(0, 0)] if not destination.collidable else World.NAV_ACTIONS
                # Maintain shortest distance.
                shortest_dist = np.inf
                for source_edge, dest_edge in product(source_edges, destination_edges):
                    try:
                        dist = nx.shortest_path_length(self.world.reachability_graph, (source.location,source_edge), (destination.location, dest_edge))
                        # Update shortest distance.
                        if dist < shortest_dist:
                            shortest_dist = dist
                    except:
                        continue
                # Cache distance floor -> counter.
                self.distances[source.location][destination.location] = shortest_dist

        # Save all distances under world as well.
        self.world.distances = self.distances
    
