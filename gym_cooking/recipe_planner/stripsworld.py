import recipe_planner.utils as recipe

# core modules
from utils.core import Object

# helpers
import networkx as nx
import copy


class STRIPSWorld:

    # PROJECT INVOLVED THIS FUNCTION CHANGE.
    def __init__(self, world, recipes):
        self.initial = recipe.STRIPSState()
        self.recipes = recipes
        self.numberOfPlates = 0

        # set initial state
        self.initial.add_predicate(recipe.NoPredicate())
        for obj in world.get_object_list():
            if isinstance(obj, Object):
                # Additional objects added
                for obj_name in ['Plate', 'Tomato', 'Lettuce', 'Onion', 'Bread', 'Cheese']:
                    if obj.contains(obj_name):
                        if obj_name == 'Plate':
                            self.numberOfPlates += 1
                        self.initial.add_predicate(recipe.Fresh(obj_name))
                for obj_name in ['Chicken', 'Fish']:
                    if obj.contains(obj_name):
                        self.initial.add_predicate(recipe.Unfried(obj_name))
                for obj_name in ['BurgerMeat']:
                    if obj.contains(obj_name):
                        self.initial.add_predicate(recipe.Uncooked(obj_name))
                for obj_name in ['PizzaDough']:
                    if obj.contains(obj_name):
                        self.initial.add_predicate(recipe.Unbaked(obj_name))

    # PROJECT INVOLVED THIS FUNCTION CHANGE.
    def generate_graph(self, recipel, max_path_length, action_path_length):
        """
        Create a graph search based on current subtasks and transitions,
        altered for plate recreation and state reset after delivery.

        Args:
            recipel: The recipe specified, name changed to avoid repetition
            max_path_length: Maximum length of the graph
            action_path_length: Current amount of action paths in the list
        Return:
            The graph generated, and the goal_state searched
        """
        all_actions = recipel.actions   # set
        goal_state = None

        new_preds = set()
        graph = nx.DiGraph()

        if action_path_length != 0:
            # Account for dirty plate possibility
            if recipe.Fresh("Plate") in self.initial.predicates:
                self.initial.delete_predicate(recipe.Fresh("Plate"))
                self.initial.add_predicate(recipe.Uncleaned("Plate"))

        graph.add_node(self.initial, obj=self.initial)
        frontier = set([self.initial])
        next_frontier = set()
        for i in range(max_path_length):
            for state in frontier:
                # for each action, check whether from this state
                for a in all_actions:
                    if a.is_valid_in(state):
                        next_state = a.get_next_from(state, self.numberOfPlates)
                        for p in next_state.predicates:
                            new_preds.add(str(p))
                        graph.add_node(next_state, obj=next_state)
                        graph.add_edge(state, next_state, obj=a)

                        # as soon as goal is found, break and return                     
                        if self.check_goal(recipel, next_state) and goal_state is None:
                            goal_state = next_state
                            return graph, goal_state
                        
                        next_frontier.add(next_state)

            frontier = next_frontier.copy()
        
        if goal_state is None:
            print('goal state could not be found, try increasing --max-num-subtasks')
            import sys; sys.exit(0)
        
        return graph, goal_state

    # PROJECT INVOLVED THIS FUNCTION CHANGE.
    def get_subtasks(self, max_path_length=10, draw_graph=False):
        action_paths = []

        for recipe in self.recipes:
            graph, goal_state = self.generate_graph(recipe, max_path_length, len(action_paths))

            if draw_graph:   # not recommended for path length > 4
                nx.draw(graph, with_labels=True)
                plt.show()
            
            all_state_paths = nx.all_shortest_paths(graph, self.initial, goal_state)
            union_action_path = set()
            for state_path in all_state_paths:
                action_path = [graph[state_path[i]][state_path[i+1]]['obj'] for i in range(len(state_path)-1)]
                union_action_path = union_action_path | set(action_path)
            # print('all tasks for recipe {}: {}\n'.format(recipe, ', '.join([str(a) for a in union_action_path])))
            action_paths.append(union_action_path)

        return action_paths
        

    def check_goal(self, recipe, state):
        # check if this state satisfies completion of this recipe
        return state.contains(recipe.goal)




