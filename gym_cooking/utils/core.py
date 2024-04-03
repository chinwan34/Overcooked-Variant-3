# recipe planning
import recipe_planner.utils as recipe
from recipe_planner.utils import Get, Chop, Deliver, Merge, Bake, Cook, Fry, Clean

# helpers
import numpy as np
import copy
import random
from termcolor import colored as color
from itertools import combinations
from collections import namedtuple

class Role:
    def __init__(self):
        self.probableActions = None

class IdlePerson(Role):
    def __init__(self):
        self.probableActions = []
        self.name = "IdlePerson"

class Merger(Role):
    def __init__(self):
        self.probableActions = [Get, Merge]
        self.name = "Merger"

class Chopper(Role):
    def __init__(self):
        self.probableActions = [Get, Chop]
        self.name = "Chopper"

class Deliverer(Role):
    def __init__(self):
        self.probableActions = [Get, Deliver, Clean]
        self.name = "Deliverer"

class Baker(Role):
    def __init__(self):
        self.probableActions = [Get, Bake]
        self.name = "Baker"

class Cooker(Role):
    def __init__(self):
        self.probableActions = [Get, Cook]
        self.name = "Cooker"

class Frier(Role):
    def __init__(self):
        self.probableActions = [Get, Fry]
        self.name = "Frier"

class Cleaner(Role):
    def __init__(self):
        self.probableActions = [Get, Clean]
        self.name = "Cleaner"
    
class MergingWaiter(Role):
    def __init__(self):
        self.probableActions = [Get, Merge, Deliver, Clean]
        self.name = "MergingWaiter"

class ChoppingMerger(Role):
    def __init__(self):
        self.probableActions = [Get, Chop, Merge]
        self.name = "ChoppingMerger"

class ChoppingWaiter(Role):
    def __init__(self):
        self.probableActions = [Get, Chop, Deliver, Clean]
        self.name = "ChoppingWaiter"

class CookingWaiter(Role):
    def __init__(self):
        self.probableActions = [Get, Cook, Deliver, Clean]
        self.name = "CookingWaiter"

class CookingMergingWaiter(Role):
    def __init__(self):
        self.probableActions = [Get, Cook, Merge, Deliver]
        self.name = "CookingMergingWaiter"

class FryingMergingWaiter(Role):
    def __init__(self):
        self.probableActions = [Get, Fry, Merge, Deliver]
        self.name = "FryingMergingWaiter"

class ExceptionalChefMerger(Role):
    def __init__(self):
        self.probableActions = [Get, Fry, Bake, Cook, Chop, Merge]
        self.name = "ExceptionalChefMerger"

class BakingWaiter(Role):
    def __init__(self):
        self.probableActions = [Get, Bake, Deliver, Clean]
        self.name = "BakingWaiter"

class FryingWaiter(Role):
    def __init__(self):
        self.probableActions = [Get, Fry, Deliver, Clean]
        self.name = "FryingWaiter"

class ExceptionalChef(Role):
    def __init__(self):
        self.probableActions = [Get, Fry, Bake, Cook, Chop]
        self.name = "ExceptionalChef"


class InvincibleWaiter(Role):
    def __init__(self):
        self.probableActions = [Get, Chop, Cook, Clean, Bake, Merge, Deliver, Fry]
        self.name = "InvincibleWaiter"



    


# -----------------------------------------------------------
# GRIDSQUARES
# -----------------------------------------------------------
GridSquareRepr = namedtuple("GridSquareRepr", "name location holding")


class Rep:
    FLOOR = ' '
    COUNTER = '-'
    CUTBOARD = '/'
    DELIVERY = '*'
    FRYER = '%'
    COOKINGPAN = '!'
    PIZZAOVEN = '?'
    TOMATO = 't'
    LETTUCE = 'l'
    ONION = 'o'
    PLATE = 'p'

    # Additional objects created for project.
    BREAD = 'b'
    BURGERMEAT = 'm'
    FISH = 'f'
    CHICKEN = 'k'
    PIZZADOUGH = 'P'
    CHEESE = 'c'
    SINK = '1'
    TRASHCAN = '$'


class GridSquare:
    def __init__(self, name, location):
        self.name = name
        self.location = location   # (x, y) tuple
        self.holding = None
        self.color = 'white'
        self.collidable = True     # cannot go through
        self.dynamic = False       # cannot move around

    def __str__(self):
        return color(self.rep, self.color)

    def __eq__(self, o):
        return isinstance(o, GridSquare) and self.name == o.name

    def __copy__(self):
        gs = type(self)(self.location)
        gs.__dict__ = self.__dict__.copy()
        if self.holding is not None:
            gs.holding = copy.copy(self.holding)
        return gs

    def acquire(self, obj):
        obj.location = self.location
        self.holding = obj

    def release(self):
        temp = self.holding
        self.holding = None
        return temp

# PROJECT INVOLVED THIS CLASS CREATION.
class TrashCan(GridSquare):
    def __init__(self, location):
        GridSquare.__init__(self, "TrashCan", location)
        self.rep = Rep.TRASHCAN
        self.colldiable = True
    def __eq__(self, other):
        return GridSquare.__eq__(self, other)
    def __hash__(self):
        return GridSquare.__hash__(self)

# PROJECT INVOLVED THIS CLASS CREATION.
class Sink(GridSquare):
    def __init__(self, location):
        GridSquare.__init__(self, "Sink", location)
        self.rep = Rep.SINK
        self.collidable = True
    def __eq__(self, other):
        return GridSquare.__eq__(self, other)
    def __hash__(self):
        return GridSquare.__hash__(self)

# PROJECT INVOLVED THIS CLASS CREATION.
class Fryer(GridSquare):
    def __init__(self, location):
        GridSquare.__init__(self, "Fryer", location)
        self.rep = Rep.FRYER
        self.collidable = True
    def __eq__(self, other):
        return GridSquare.__eq__(self, other)
    def __hash__(self):
        return GridSquare.__hash__(self)

# PROJECT INVOLVED THIS CLASS CREATION.
class PizzaOven(GridSquare):
    def __init__(self, location):
        GridSquare.__init__(self, "PizzaOven", location)
        self.rep = Rep.PIZZAOVEN
        self.collidable = True
    def __eq__(self, other):
        return GridSquare.__eq__(self, other)
    def __hash__(self):
        return GridSquare.__hash__(self)

# PROJECT INVOLVED THIS CLASS CREATION.
class CookingPan(GridSquare):
    def __init__(self, location):
        GridSquare.__init__(self, "CookingPan", location)
        self.rep = Rep.COOKINGPAN
        self.collidable = True
    def __eq__(self, other):
        return GridSquare.__eq__(self, other)
    def __hash__(self):
        return GridSquare.__hash__(self)

class Floor(GridSquare):
    def __init__(self, location):
        GridSquare.__init__(self,"Floor", location)
        self.color = None
        self.rep = Rep.FLOOR
        self.collidable = False
    def __eq__(self, other):
        return GridSquare.__eq__(self, other)
    def __hash__(self):
        return GridSquare.__hash__(self)

class Counter(GridSquare):
    def __init__(self, location):
        GridSquare.__init__(self,"Counter", location)
        self.rep = Rep.COUNTER
    def __eq__(self, other):
        return GridSquare.__eq__(self, other)
    def __hash__(self):
        return GridSquare.__hash__(self)

class AgentCounter(Counter):
    def __init__(self, location):
        GridSquare.__init__(self,"Agent-Counter", location)
        self.rep = Rep.COUNTER
        self.collidable = True
        # self.name = "AgentCounter"
    def __eq__(self, other):
        return Counter.__eq__(self, other)
    def __hash__(self):
        return Counter.__hash__(self)
    def get_repr(self):
        return GridSquareRepr(name=self.name, location=self.location, holding= None)

class Cutboard(GridSquare):
    def __init__(self, location):
        GridSquare.__init__(self, "Cutboard", location)
        self.rep = Rep.CUTBOARD
        self.collidable = True
    def __eq__(self, other):
        return GridSquare.__eq__(self, other)
    def __hash__(self):
        return GridSquare.__hash__(self)

class Delivery(GridSquare):
    def __init__(self, location):
        GridSquare.__init__(self, "Delivery", location)
        self.rep = Rep.DELIVERY
        self.holding = []
    def acquire(self, obj):
        obj.location = self.location
        self.holding.append(obj)
    def release(self):
        if self.holding:
            return self.holding.pop()
        else: return None
    def __eq__(self, other):
        return GridSquare.__eq__(self, other)
    def __hash__(self):
        return GridSquare.__hash__(self)


# -----------------------------------------------------------
# OBJECTS
# -----------------------------------------------------------
# Objects are wrappers around foods items, plates, and any combination of them

ObjectRepr = namedtuple("ObjectRepr", "name location is_held")

class Object:
    def __init__(self, location, contents):
        self.location = location
        self.contents = contents if isinstance(contents, list) else [contents]
        self.is_held = False
        self.update_names()
        self.collidable = False
        self.dynamic = False

    def __str__(self):
        res = "-".join(list(map(lambda x : str(x), sorted(self.contents, key=lambda i: i.name))))
        return res

    def __eq__(self, other):
        # check that content is the same and in the same state(s)
        return isinstance(other, Object) and \
                self.name == other.name and \
                len(self.contents) == len(other.contents) and \
                self.full_name == other.full_name
                # all([i == j for i, j in zip(sorted(self.contents, key=lambda x: x.name),
                #                             sorted(other.contents, key=lambda x: x.name))])

    def __copy__(self):
        new = Object(self.location, self.contents[0])
        new.__dict__ = self.__dict__.copy()
        new.contents = [copy.copy(c) for c in self.contents]
        return new
    
    def returnContents(self):
        return self.contents

    def get_repr(self):
        return ObjectRepr(name=self.full_name, location=self.location, is_held=self.is_held)

    def update_names(self):
        # concatenate names of alphabetically sorted items, e.g.
        sorted_contents = sorted(self.contents, key=lambda c: c.name)
        self.name = "-".join([c.name for c in sorted_contents])
        self.full_name = "-".join([c.full_name for c in sorted_contents])

    def contains(self, c_name):
        return c_name in list(map(lambda c : c.name, self.contents))

    # Auxiliary function to check whether contents needs to be done with actions
    # PROJECT INVOLVED THESE FUNCTIONS CHANGE.
    def needs_chopped(self):
        if len(self.contents) > 1: return False
        return self.contents[0].needs_chopped()

    def needs_cleaned(self):
        if len(self.contents) > 1: return False
        return self.contents[0].needs_cleaned()
    
    def needs_cooked(self):
        if len(self.contents) > 1: return False
        return self.contents[0].needs_cooked()

    def needs_fried(self):
        if len(self.contents) > 1:return False
        return self.contents[0].needs_fried()

    def needs_baked(self):
        if len(self.contents) > 1: return False
        return self.contents[0].needs_baked()

    def is_chopped(self):
        for c in self.contents:
            if isinstance(c, Plate) or c.get_state() != 'Chopped':
                return False
        return True

    def chop(self):
        assert len(self.contents) == 1
        assert self.needs_chopped()
        self.contents[0].update_state()
        assert not (self.needs_chopped())
        self.update_names()
    
    def cook(self):
        assert len(self.contents) == 1
        assert self.needs_cooked()
        self.contents[0].update_state()
        assert not (self.needs_cooked())
        self.update_names()
    
    def bake(self):
        # There may be issue here later
        assert self.contents[0].needs_baked()
        self.contents[0].update_state()
        assert not (self.contents[0].needs_baked())
        self.update_names()
    
    def fry(self):
        assert len(self.contents) == 1
        assert self.needs_fried()
        self.contents[0].update_state()
        assert not (self.needs_fried())
        self.update_names() 
    
    def clean(self):
        assert len(self.contents) == 1
        assert self.needs_cleaned()
        self.contents[0].update_state()
        self.update_names()


    def merge(self, obj):
        if isinstance(obj, Object):
            # move obj's contents into this instance
            for i in obj.contents: self.contents.append(i)
        elif not (isinstance(obj, Food) or isinstance(obj, Plate)):
            raise ValueError("Incorrect merge object: {}".format(obj))
        else:
            self.contents.append(obj)
        self.update_names()

    def unmerge(self, full_name):
        # remove by full_name, assumming all unique contents
        matching = list(filter(lambda c: c.full_name == full_name, self.contents))
        self.contents.remove(matching[0])
        self.update_names()
        return matching[0]

    def is_merged(self):
        return len(self.contents) > 1

    def is_deliverable(self):
        # must be merged, and all contents must be Plates or Foods in done state
        for c in self.contents:
            if not (isinstance(c, Plate) or (isinstance(c, Food) and c.done())):
                return False
        return self.is_merged()


# PROJECT INVOLVED THIS FUNCTION CHANGE.
def mergeable(obj1, obj2):
    # query whether two objects are mergeable
    contents = obj1.contents + obj2.contents
    # check that there is at most one plate
    for c in contents:
        if not c.done():
            return False
    try:
        contents.remove(Plate())
    except:
        pass  # do nothing, 1 plate is ok
    finally:
        try:
            contents.remove(Plate())
        except:
            for c in contents:   # everything else must be in last state
                if not c.done():
                    return False
        else:
            return False  # more than 1 plate
    return True


# -----------------------------------------------------------

class FoodState:
    FRESH = globals()['recipe'].__dict__['Fresh']
    CHOPPED = globals()['recipe'].__dict__['Chopped']
    UNCOOKED = globals()['recipe'].__dict__['Uncooked']
    UNFRIED = globals()['recipe'].__dict__['Unfried']
    UNBAKED = globals()['recipe'].__dict__['Unbaked']
    COOKED = globals()['recipe'].__dict__['Cooked']
    MERGED = globals()['recipe'].__dict__['Merged']
    UNCLEANED = globals()['recipe'].__dict__['Uncleaned']

# PROJECT INVOLVED THIS CLASS CHANGE.
class FoodSequence:
    UNCLEANED_FRESH = [FoodState.UNCLEANED, FoodState.FRESH]
    FRESH = [FoodState.FRESH]
    FRESH_CHOPPED = [FoodState.FRESH, FoodState.CHOPPED]
    UNCOOKED_COOKED = [FoodState.UNCOOKED, FoodState.COOKED]
    UNFRIED_COOKED = [FoodState.UNFRIED, FoodState.COOKED]
    UNBAKED_COOKED = [FoodState.UNBAKED, FoodState.COOKED]

class Food:
    def __init__(self):
        self.state = self.state_seq[self.state_index]
        self.movable = False
        self.color = self._set_color()
        self.update_names()

    def __str__(self):
        return color(self.rep, self.color)

    # def __hash__(self):
    #     return hash((self.state, self.name))

    def __eq__(self, other):
        return isinstance(other, Food) and self.get_state() == other.get_state()

    def __len__(self):
        return 1   # one food unit

    def set_state(self, state):
        assert state in self.state_seq, "Desired state {} does not exist for the food with sequence {}".format(state, self.state_seq)
        self.state_index = self.state_seq.index(state)
        self.state = state
        self.update_names()

    def get_state(self):
        return self.state.__name__

    def update_names(self):
        self.full_name = '{}{}'.format(self.get_state(), self.name)
    
    # Auxiliary objects to overwrite the object methods.
    # PROJECT INVOLVED THESE FUNCTIONS CHANGE.
    def needs_cleaned(self):
        return self.state_seq[(self.state_index)%len(self.state_seq)] == FoodState.UNCLEANED

    def needs_chopped(self):
        return self.state_seq[(self.state_index+1)%len(self.state_seq)] == FoodState.CHOPPED
    
    def needs_cooked(self):
        return self.state_seq[(self.state_index+1)%len(self.state_seq)] == FoodState.COOKED and (self.state_seq[self.state_index] == FoodState.UNCOOKED)
    
    def needs_fried(self):
        return (self.state_seq[(self.state_index+1)%len(self.state_seq)] == FoodState.COOKED) and (self.state_seq[self.state_index] == FoodState.UNFRIED)

    def needs_baked(self):
        return (self.state_seq[(self.state_index+1)%len(self.state_seq)] == FoodState.COOKED) and (self.state_seq[self.state_index] == FoodState.UNBAKED)

    def done(self):
        return (self.state_index % len(self.state_seq)) == len(self.state_seq) - 1

    def update_state(self):
        self.state_index += 1
        assert 0 <= self.state_index and self.state_index < len(self.state_seq), "State index is out of bounds for its state sequence"
        self.state = self.state_seq[self.state_index]
        self.update_names()

    def _set_color(self):
        pass

# PROJECT INVOLVED THIS CLASS CREATION.
class BurgerMeat(Food):
    def __init__(self, state_index=0):
        self.state_index = state_index
        self.state_seq = FoodSequence.UNCOOKED_COOKED
        self.rep = 'm'
        self.name = 'BurgerMeat'
        Food.__init__(self)
    def __hash__(self):
        return Food.__hash__(self)
    def __eq__(self, other):
        return Food.__eq__(self, other)
    def __str__(self):
        return Food.__str__(self)
    def needs_chopped(self):
        return False

# PROJECT INVOLVED THIS CLASS CREATION.
class Bread(Food):
    def __init__(self, state_index=0):
        self.state_index = state_index
        self.state_seq = FoodSequence.FRESH
        self.rep = 'b'
        self.name = 'Bread'
        Food.__init__(self)
    def __hash__(self):
        return Food.__hash__(self)
    def __eq__(self, other):
        return Food.__eq__(self, other)
    def __str__(self):
        return Food.__str__(self)
    def needs_chopped(self):
        return False

# PROJECT INVOLVED THIS CLASS CREATION.
class PizzaDough(Food):
    def __init__(self, state_index=0):
        self.state_index = state_index
        self.state_seq = FoodSequence.UNBAKED_COOKED
        self.rep = 'p'
        self.name = 'PizzaDough'
        Food.__init__(self)
    def __hash__(self):
        return Food.__hash__(self)
    def __eq__(self, other):
        return Food.__eq__(self, other)
    def __str__(self):
        return Food.__str__(self)
    def needs_chopped(self):
        return False

# PROJECT INVOLVED THIS CLASS CREATION.
class Cheese(Food):
    def __init__(self, state_index=0):
        self.state_index = state_index
        self.state_seq = FoodSequence.FRESH_CHOPPED
        self.rep = 'c'
        self.name = 'Cheese'
        Food.__init__(self)
    def __hash__(self):
        return Food.__hash__(self)
    def __eq__(self, other):
        return Food.__eq__(self, other)
    def __str__(self):
        return Food.__str__(self)

# PROJECT INVOLVED THIS CLASS CREATION.
class FriedChicken(Food):
    def __init__(self, state_index = 0):
        state_index = 0
        self.state_index = state_index
        self.state_seq = FoodSequence.UNFRIED_COOKED
        self.rep = 'k'
        self.name = 'Chicken'
        Food.__init__(self)
    def __hash__(self):
        return Food.__hash__(self)
    def __eq__(self, other):
        return Food.__eq__(self, other)
    def __str__(self):
        return Food.__str__(self)

# PROJECT INVOLVED THIS CLASS CREATION.
class Fish(Food):
    def __init__(self, state_index=0):
        state_index = 0
        self.state_index = state_index
        self.state_seq = FoodSequence.UNFRIED_COOKED
        self.rep = 'f'
        self.name = 'Fish'
        Food.__init__(self)
    def __hash__(self):
        return Food.__hash__(self)
    def __eq__(self, other):
        return Food.__eq__(self, other)
    def __str__(self):
        return Food.__str__(self)
    def needs_chopped(self):
        return False

class Tomato(Food):
    def __init__(self, state_index = 0):
        self.state_index = state_index   # index in food's state sequence
        self.state_seq = FoodSequence.FRESH_CHOPPED
        self.rep = 't'
        self.name = 'Tomato'
        Food.__init__(self)
    def __hash__(self):
        return Food.__hash__(self)
    def __eq__(self, other):
        return Food.__eq__(self, other)
    def __str__(self):
        return Food.__str__(self)

class Lettuce(Food):
    def __init__(self, state_index = 0):
        self.state_index = state_index   # index in food's state sequence
        self.state_seq = FoodSequence.FRESH_CHOPPED
        self.rep = 'l'
        self.name = 'Lettuce'
        Food.__init__(self)
    def __eq__(self, other):
        return Food.__eq__(self, other)
    def __hash__(self):
        return Food.__hash__(self)

class Onion(Food):
    def __init__(self, state_index = 0):
        self.state_index = state_index   # index in food's state sequence
        self.state_seq = FoodSequence.FRESH_CHOPPED
        self.rep = 'o'
        self.name = 'Onion'
        Food.__init__(self)
    def __eq__(self, other):
        return Food.__eq__(self, other)
    def __hash__(self):
        return Food.__hash__(self)


# -----------------------------------------------------------

class Plate:
    def __init__(self, state_index = 1):
        self.state_index = state_index
        self.state_seq = FoodSequence.UNCLEANED_FRESH
        self.rep = "p"
        self.name = 'Plate'
        self.full_name = 'Plate'
        self.color = 'white'
    
    def done(self):
        return (self.state_index % len(self.state_seq)) == len(self.state_seq) - 1
    
    # Following functions update states and names of plate based on
    # The reverse index [Dirty, Clean] representation.
    # PROJECT INVOLVED THESE FUNCTIONS CREATION.
    def update_names(self):
        self.full_name = "Plate"
    
    def update_dirty_name(self):
        self.full_name = "Dirty-Plate"
    
    def update_state(self):
        self.state_index += 1
        assert 0 <= self.state_index and self.state_index < len(self.state_seq), "State index is out of bounds for its state sequence"
        self.state = self.state_seq[self.state_index]
        self.update_names

    def needs_cleaned(self):
        return self.state_seq[(self.state_index)%len(self.state_seq)] == FoodState.UNCLEANED
    def __hash__(self):
        return hash((self.name))
    def __str__(self):
        return color(self.rep, self.color)
    def __eq__(self, other):
        return isinstance(other, Plate)
    def __copy__(self):
        return Plate()
    def needs_chopped(self):
        return False
    def needs_fried(self):
        return False
    def needs_cooked(self):
        return False
    def needs_baked(self):
        return False


# -----------------------------------------------------------
# PARSING
# -----------------------------------------------------------
RepToClass = {
    Rep.FLOOR: globals()['Floor'],
    Rep.COUNTER: globals()['Counter'],
    Rep.CUTBOARD: globals()['Cutboard'],
    Rep.DELIVERY: globals()['Delivery'],
    Rep.TOMATO: globals()['Tomato'],
    Rep.LETTUCE: globals()['Lettuce'],
    Rep.ONION: globals()['Onion'],
    Rep.PLATE: globals()['Plate'],
    Rep.CHICKEN: globals()['FriedChicken'],
    Rep.FISH: globals()['Fish'],
    Rep.BREAD: globals()['Bread'],
    Rep.BURGERMEAT: globals()['BurgerMeat'],
    Rep.FRYER: globals()['Fryer'],
    Rep.COOKINGPAN: globals()['CookingPan'],
    Rep.CHEESE: globals()['Cheese'],
    Rep.PIZZADOUGH: globals()['PizzaDough'],
    Rep.PIZZAOVEN: globals()['PizzaOven'],
    Rep.SINK: globals()['Sink'],
    Rep.TRASHCAN: globals()['TrashCan'],
}



