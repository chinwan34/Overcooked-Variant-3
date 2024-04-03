from utils.core import *
import recipe_planner.utils as recipe
from itertools import permutations


class Recipe:
    def __init__(self, name):
        self.name = name
        self.contents = []
        self.actions = set()
        self.actions.add(recipe.Get('Plate'))

    def __str__(self):
        return self.name

    # PROJECT INVOLVED THIS FUNCTION CHANGE.
    def add_ingredient(self, item):
        """
        Add the actions that are part of the item for subtask
        search. Additional subtasks and actions added below.

        Args:
            item: The ingredient specified
        """
        self.contents.append(item)

        # always starts with FRESH
        self.actions.add(recipe.Get(item.name))

        if item.state_seq == FoodSequence.FRESH_CHOPPED:
            self.actions.add(recipe.Chop(item.name))
            self.actions.add(recipe.Merge(item.name, 'Plate',\
                [item.state_seq[-1](item.name), recipe.Fresh('Plate')], None))
        
        # Additional FoodSequence created
        if item.state_seq == FoodSequence.UNFRIED_COOKED:
            self.actions.add(recipe.Fry(item.name))
            self.actions.add(recipe.Merge(item.name, 'Plate',\
                [item.state_seq[-1](item.name), recipe.Fresh('Plate')], None))
        
        if item.state_seq == FoodSequence.UNCOOKED_COOKED:
            self.actions.add(recipe.Cook(item.name))
            self.actions.add(recipe.Merge(item.name, 'Plate',\
                [item.state_seq[-1](item.name), recipe.Fresh('Plate')], None))
        
        if item.state_seq == FoodSequence.UNBAKED_COOKED:
            self.actions.add(recipe.Bake(item.name))
            self.actions.add(recipe.Merge(item.name, 'Plate',\
                [item.state_seq[-1](item.name), recipe.Fresh('Plate')], None))  
    
    # PROJECT INVOLVED THIS FUNCTION CHANGE.
    def add_uncleaned_plates_issue(self):
        """
        Included the uncleaned plates possibility in actions,
        only available in manual play mode.
        """
        self.actions.add(recipe.Clean(Plate().full_name)) 

    def add_goal(self):
        self.contents = sorted(self.contents, key = lambda x: x.name)   # list of Food objects
        self.contents_names = [c.name for c in self.contents]   # list of strings
        self.full_name = '-'.join(sorted(self.contents_names))   # string
        self.full_plate_name = '-'.join(sorted(self.contents_names + ['Plate']))   # string
        self.goal = recipe.Delivered(self.full_plate_name)
        self.actions.add(recipe.Deliver(self.full_plate_name))

    # PROJECT INVOLVED THIS FUNCTION CHANGE.
    def add_merge_actions(self):
        """
        Add the possible merging action between
        multiple ingredients. Considered multiple ingredient-burger
        for complex merging actions possibilities.
        """
        # should be general enough for any kind of salad / raw plated veggies
        isBurger = False
        types = [('Tomato', recipe.Chopped), ('Lettuce', recipe.Chopped), ('BurgerMeat', recipe.Cooked), ('Bread', recipe.Fresh)]
        type_permutations = list(permutations(types, 2))
        for i in self.contents:
            if isinstance(i, BurgerMeat):
                isBurger = True

        # alphabetical, joined by dashes ex. Ingredient1-Ingredient2-Plate
        #self.full_name = '-'.join(sorted(self.contents + ['Plate']))
        # for any plural number of ingredients
        for i in range(2, len(self.contents)+1):
            # for any combo of i ingredients to be merged
            for combo in combinations(self.contents_names, i):
                # can merge all with plate
                self.actions.add(recipe.Merge('-'.join(sorted(combo)), 'Plate',\
                    [recipe.Merged('-'.join(sorted(combo))), recipe.Fresh('Plate')], None))

                # for any one item to be added to the i-1 rest
                for item in combo:
                    rem = list(combo).copy()
                    rem.remove(item)
                    rem_str = '-'.join(sorted(rem))
                    plate_str = '-'.join(sorted([item, 'Plate']))
                    rem_plate_str = '-'.join(sorted(rem + ['Plate']))

                    # can merge item with remaining
                    if isBurger:
                        # Deal with burger merging issues
                        if len(rem) == 1:
                            for i,j in type_permutations:
                                if item == i[0] and rem[0] == j[0]:
                                    self.actions.add(recipe.Merge(item, rem_str,\
                                        [i[1](item), j[1](rem_str)], None))
                                    self.actions.add(recipe.Merge(rem_str, plate_str))
                                    self.actions.add(recipe.Merge(item, rem_plate_str))
                                    break
                        else:
                            self.actions.add(recipe.Merge(item, rem_str))
                            self.actions.add(recipe.Merge(plate_str, rem_str,\
                                [recipe.Merged(plate_str), recipe.Merged(rem_str)], None))
                            self.actions.add(recipe.Merge(item, rem_plate_str))
                    else:
                        if len(rem) == 1:
                            self.actions.add(recipe.Merge(item, rem_str,\
                                [recipe.Chopped(item), recipe.Chopped(rem_str)], None))
                            self.actions.add(recipe.Merge(rem_str, plate_str))
                            self.actions.add(recipe.Merge(item, rem_plate_str))
                        else:
                            self.actions.add(recipe.Merge(item, rem_str))
                            self.actions.add(recipe.Merge(plate_str, rem_str,\
                                [recipe.Merged(plate_str), recipe.Merged(rem_str)], None))
                            self.actions.add(recipe.Merge(item, rem_plate_str))

# PROJECT INVOLVED THIS CLASS CREATION.
class FriedChickenRe(Recipe):
    def __init__(self):
        Recipe.__init__(self, 'FriedChickenRe')
        self.add_ingredient(FriedChicken(state_index=-1))
        self.add_goal()
        self.add_merge_actions()
        self.add_uncleaned_plates_issue()

# PROJECT INVOLVED THIS CLASS CREATION.
class SimplePizza(Recipe):
    def __init__(self):
        Recipe.__init__(self, 'SimplePizza')
        self.add_ingredient(PizzaDough(state_index=-1))
        self.add_ingredient(Cheese(state_index=-1))
        self.add_goal()
        self.add_merge_actions()
        self.add_uncleaned_plates_issue()
    
    def add_merge_actions(self):
         """
         Overwriting to superclass, with its own 
         combination of merging action due to unique ingredients.
         """
         for i in range(2, len(self.contents)+1):
            # for any combo of i ingredients to be merged
            for combo in combinations(self.contents_names, i):
                # can merge all with plate
                self.actions.add(recipe.Merge('-'.join(sorted(combo)), 'Plate',\
                    [recipe.Merged('-'.join(sorted(combo))), recipe.Fresh('Plate')], None))

                # for any one item to be added to the i-1 rest
                for item in combo:
                    rem = list(combo).copy()
                    rem.remove(item)
                    rem_str = '-'.join(sorted(rem))
                    plate_str = '-'.join(sorted([item, 'Plate']))
                    rem_plate_str = '-'.join(sorted(rem + ['Plate']))

                    # can merge item with remaining
                    # consider simply for cheese and pizzadough object
                    if len(rem) == 1:
                        if item == "PizzaDough" and rem[0] == "Cheese":
                            self.actions.add(recipe.Merge(item, rem_str,\
                                [recipe.Cooked(item), recipe.Chopped(rem_str)], None))
                        elif item == "Cheese" and rem[0] == "PizzaDough":
                            self.actions.add(recipe.Merge(item, rem_str,\
                                [recipe.Chopped(item), recipe.Cooked(rem_str)], None))
                        self.actions.add(recipe.Merge(rem_str, plate_str))
                        self.actions.add(recipe.Merge(item, rem_plate_str))
                    else:
                        self.actions.add(recipe.Merge(item, rem_str))
                        self.actions.add(recipe.Merge(plate_str, rem_str,\
                            [recipe.Merged(plate_str), recipe.Merged(rem_str)], None))
                        self.actions.add(recipe.Merge(item, rem_plate_str))

# PROJECT INVOLVED THIS CLASS CREATION.
class FriedFishRe(Recipe):
    def __init__(self):
        Recipe.__init__(self, 'FriedFishRe')
        self.add_ingredient(Fish(state_index=-1))
        self.add_goal()
        self.add_merge_actions()
    
# PROJECT INVOLVED THIS CLASS CREATION.
class FishAndChicken(Recipe):
    def __init__(self):
        Recipe.__init__(self, 'FishAndChicken')
        self.add_ingredient(Fish(state_index=-1))
        self.add_ingredient(FriedChicken(state_index=-1))
        self.add_goal()
        self.add_merge_actions()
        self.add_uncleaned_plates_issue()
    
    def add_merge_actions(self):
         """
         Overwriting to superclass for the fish and chicken, with its own 
         combination of merging action due to unique ingredients.
         """
         for i in range(2, len(self.contents)+1):
            # for any combo of i ingredients to be merged
            for combo in combinations(self.contents_names, i):
                # can merge all with plate
                self.actions.add(recipe.Merge('-'.join(sorted(combo)), 'Plate',\
                    [recipe.Merged('-'.join(sorted(combo))), recipe.Fresh('Plate')], None))

                # for any one item to be added to the i-1 rest
                for item in combo:
                    rem = list(combo).copy()
                    rem.remove(item)
                    rem_str = '-'.join(sorted(rem))
                    plate_str = '-'.join(sorted([item, 'Plate']))
                    rem_plate_str = '-'.join(sorted(rem + ['Plate']))

                    # can merge item with remaining
                    if len(rem) == 1:
                        self.actions.add(recipe.Merge(item, rem_str,\
                            [recipe.Cooked(item), recipe.Cooked(rem_str)], None))
                        self.actions.add(recipe.Merge(rem_str, plate_str))
                        self.actions.add(recipe.Merge(item, rem_plate_str))
                    else:
                        self.actions.add(recipe.Merge(item, rem_str))
                        self.actions.add(recipe.Merge(plate_str, rem_str,\
                            [recipe.Merged(plate_str), recipe.Merged(rem_str)], None))
                        self.actions.add(recipe.Merge(item, rem_plate_str))

# PROJECT INVOLVED THIS CLASS CREATION.
class SimpleBurger(Recipe):
    def __init__(self):
        Recipe.__init__(self, 'SimpleBurger')
        self.add_ingredient(BurgerMeat(state_index=-1))
        self.add_ingredient(Bread(state_index=-1))
        self.add_goal()
        self.add_merge_actions()
        self.add_uncleaned_plates_issue()

# PROJECT INVOLVED THIS CLASS CREATION.
class LettuceBurger(Recipe):
    def __init__(self):
        Recipe.__init__(self, 'LettuceBurger')
        self.add_ingredient(BurgerMeat(state_index=-1))
        self.add_ingredient(Bread(state_index=-1))
        self.add_ingredient(Lettuce(state_index=-1))
        self.add_goal()
        self.add_merge_actions()
        self.add_uncleaned_plates_issue()

# PROJECT INVOLVED THIS CLASS CREATION.
class TomatoBurger(Recipe):
    def __init__(self):
        Recipe.__init__(self, 'TomatoBurger')
        self.add_ingredient(BurgerMeat(state_index=-1))
        self.add_ingredient(Bread(state_index=-1))
        self.add_ingredient(Tomato(state_index=-1))
        self.add_goal()
        self.add_merge_actions()
        self.add_uncleaned_plates_issue()

# PROJECT INVOLVED THIS CLASS CREATION.
class SaladBurger(Recipe):
    def __init__(self):
        Recipe.__init__(self, 'SaladBurger')
        self.add_ingredient(BurgerMeat(state_index=-1))
        self.add_ingredient(Bread(state_index=-1))
        self.add_ingredient(Tomato(state_index=-1))
        self.add_ingredient(Lettuce(state_index=-1))
        self.add_goal()
        self.add_merge_actions()
        self.add_uncleaned_plates_issue()

class SimpleTomato(Recipe):
    def __init__(self):
        Recipe.__init__(self, 'Tomato')
        self.add_ingredient(Tomato(state_index=-1))
        self.add_goal()
        self.add_merge_actions()
        self.add_uncleaned_plates_issue()

class SimpleLettuce(Recipe):
    def __init__(self):
        Recipe.__init__(self, 'Lettuce')
        self.add_ingredient(Lettuce(state_index=-1))
        self.add_goal()
        self.add_merge_actions()
        self.add_uncleaned_plates_issue()

class Salad(Recipe):
    def __init__(self):
        Recipe.__init__(self, 'Salad')
        self.add_ingredient(Tomato(state_index=-1))
        self.add_ingredient(Lettuce(state_index=-1))
        self.add_goal()
        self.add_merge_actions()
        self.add_uncleaned_plates_issue()

class OnionSalad(Recipe):
    def __init__(self):
        Recipe.__init__(self, 'OnionSalad')
        self.add_ingredient(Tomato(state_index=-1))
        self.add_ingredient(Lettuce(state_index=-1))
        self.add_ingredient(Onion(state_index=-1))
        self.add_goal()
        self.add_merge_actions()
        self.add_uncleaned_plates_issue()


