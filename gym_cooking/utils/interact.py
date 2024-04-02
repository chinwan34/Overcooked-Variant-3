from utils.core import *
import numpy as np
import time

def interact(agent, world):
    """
    Cause interaction between agents and the world representation,
    this involved all possibilities of object state change, 
    agent movements, trash can disposal, and more.

    Args:
        agent: The agent to interact with the environment
        world: The representation that stores objects
    """
    from misc.game.game import Game
    """Carries out interaction for this agent taking this action in this world.

    The action that needs to be executed is stored in `agent.action`.
    """

    # agent does nothing (i.e. no arrow key)
    if agent.action == (0, 0):
        return

    action_x, action_y = world.inbounds(tuple(np.asarray(agent.location) + np.asarray(agent.action)))
    gs = world.get_gridsquare_at((action_x, action_y))

    # if floor in front --> move to that square
    if isinstance(gs, Floor) and gs.holding is None:
        agent.move_to(gs.location)

    # if holding something
    elif agent.holding is not None:
        # if delivery in front --> deliver
        if isinstance(gs, Delivery) and (Deliver in agent.role.probableActions):
            obj = agent.holding
            if obj.is_deliverable():
                gs.acquire(obj)
                # world.insert(obj)
                agent.release()
                print('\nDelivered {}!'.format(obj.full_name))

                # Put back plates into the environment, dirty.
                for i in range(0, len(Game.plate_location)):
                    (currentX,currentY) = Game.plate_location[i]
                    gsPlate = world.get_gridsquare_at((currentX, currentY))
                    if gsPlate.holding is None:
                        gsPlate.acquire(Object(location=Game.plate_location[i], contents=RepToClass["p"](state_index=0)))
                        world.insert(gsPlate.holding)
                        break
        
        elif isinstance(gs, Delivery):
            pass

        # if occupied gridsquare in front --> try merging
        elif world.is_occupied(gs.location):
            # Get object on gridsquare/counter
            obj = world.get_object_at(gs.location, None, find_held_objects = False)

            if mergeable(agent.holding, obj) and (Merge in agent.role.probableActions):
                world.remove(obj)
                o = gs.release() # agent is holding object
                world.remove(agent.holding)
                agent.acquire(obj)
                world.insert(agent.holding)
                gs.acquire(agent.holding)
                agent.release()

        # if holding something, empty gridsquare in front --> chop, cook, bake or drop
        elif not world.is_occupied(gs.location):
            obj = agent.holding
            if not isinstance(gs, Delivery):
                gs.acquire(obj)
                agent.release()
            if isinstance(gs, Cutboard) and obj.needs_chopped() and not world.arglist.play: 
                obj.chop()
            elif isinstance(gs, Fryer) and obj.needs_fried() and not world.arglist.play: 
                obj.fry()
            elif isinstance(gs, CookingPan) and obj.needs_cooked() and not world.arglist.play: 
                obj.cook()
            elif isinstance(gs, PizzaOven) and obj.needs_baked() and not world.arglist.play: 
                obj.bake()
            elif isinstance(gs, Sink) and obj.needs_cleaned() and not world.arglist.play: 
                obj.clean()
            else:
                assert world.get_object_at(gs.location, obj, find_held_objects =\
                    False).is_held == False, "Verifying put down works"

    # if not holding anything
    elif agent.holding is None:
        # not empty in front --> pick up
        if world.is_occupied(gs.location) and not isinstance(gs, Delivery):
            obj = world.get_object_at(gs.location, None, find_held_objects = False)
            # if in playable game mode, then chop raw items on cutting board, or does other actions with the same results
            if isinstance(gs, Cutboard) and obj.needs_chopped() and world.arglist.play and Chop in agent.role.probableActions:
                obj.chop()
            elif isinstance(gs, Fryer) and obj.needs_fried() and world.arglist.play and Fry in agent.role.probableActions:
                obj.fry()
            elif isinstance(gs, CookingPan) and obj.needs_cooked() and world.arglist.play and Cook in agent.role.probableActions:
                obj.cook()
            elif isinstance(gs, PizzaOven) and obj.needs_baked() and world.arglist.play and Bake in agent.role.probableActions:
                obj.bake()
            elif isinstance(gs, Sink) and obj.needs_cleaned() and world.arglist.play and Clean in agent.role.probableActions:
                obj.clean()
            # The trashcan representation and creation
            elif isinstance(gs, TrashCan) and world.arglist.play:
                alphabetClassPair = [(Fish, 'f'), (FriedChicken, 'k'), (BurgerMeat, 'm'),
                         (PizzaDough, 'P'), (Cheese, 'c'), (Bread, 'b'), (Onion, 'o'),
                         (Lettuce, 'l'), (Tomato, 't'), (Plate, 'p')]
                
                emptyNewContentsWithString = []
                for i in range(len(obj.contents)):
                    for j in range(len(alphabetClassPair)):
                        if type(obj.contents[i]) == alphabetClassPair[j][0]:
                            emptyNewContentsWithString.append(alphabetClassPair[j][1])

                # Append the plates and other objects back to original locations
                for i in range(0, len(emptyNewContentsWithString)):
                    for j in range(0, len(Game.food_locations)):
                        if Game.food_locations[j][0] == emptyNewContentsWithString[i]:
                            (currentX,currentY) = Game.food_locations[j][1]
                            gsPlate = world.get_gridsquare_at((currentX, currentY))
                            if gsPlate.holding is None:
                                if Game.food_locations[j][0] == 'p':
                                    gsPlate.acquire(Object(location=Game.food_locations[j][1], contents=RepToClass[Game.food_locations[j][0]](state_index=1)))
                                else:
                                    gsPlate.acquire(Object(location=Game.food_locations[j][1], contents=RepToClass[Game.food_locations[j][0]](state_index=0)))
                                world.insert(gsPlate.holding)
                                break
                            else:
                                continue
                world.remove(obj)

            else:
                if not isinstance(gs, Delivery):
                    gs.release()
                    agent.acquire(obj)

        # if empty in front --> interact
        elif not world.is_occupied(gs.location):
            pass
