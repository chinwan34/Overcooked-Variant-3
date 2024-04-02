import os
import pygame
import numpy as np
from utils.core import *
from misc.game.utils import *

graphics_dir = 'misc/game/graphics'
_image_library = {}

def get_image(path):
    global _image_library
    image = _image_library.get(path)
    if image == None:
        canonicalized_path = path.replace('/', os.sep).replace('\\', os.sep)
        image = pygame.image.load(canonicalized_path)
        _image_library[path] = image
    return image


class Game:
    plate_location = []
    food_locations = []
    gridsquare_locations = []

    def __init__(self, world, sim_agents, play=False):
        self._running = True
        self.world = world
        self.sim_agents = sim_agents
        self.current_agent = self.sim_agents[0]
        self.play = play
        
        # Visual parameters
        self.scale = 80   # num pixels per tile
        self.holding_scale = 0.5
        self.container_scale = 0.7
        self.width = self.scale * self.world.width
        self.height = self.scale * self.world.height
        self.tile_size = (self.scale, self.scale)
        self.holding_size = tuple((self.holding_scale * np.asarray(self.tile_size)).astype(int))
        self.container_size = tuple((self.container_scale * np.asarray(self.tile_size)).astype(int))
        self.holding_container_size = tuple((self.container_scale * np.asarray(self.holding_size)).astype(int))
        #self.font = pygame.font.SysFont('arialttf', 10)

        self.get_plate_location()
        self.get_all_food_plate_location()
        
        

    def on_init(self):
        pygame.init()
        if self.play:
            self.screen = pygame.display.set_mode((self.width, self.height))
        else:
            # Create a hidden surface
            self.screen = pygame.Surface((self.width, self.height))
        self._running = True


    def on_event(self, event):
        if event.type == pygame.QUIT:
            self._running = False


    def on_render(self):
        self.screen.fill(Color.FLOOR)
        objs = []
        
        # Draw gridsquares
        for o_list in self.world.objects.values():
            for o in o_list:
                if isinstance(o, GridSquare):
                    self.draw_gridsquare(o)
                elif o.is_held == False:
                    objs.append(o)
        
        # Draw objects not held by agents
        for o in objs:
            self.draw_object(o)

        # Draw agents and their holdings
        for agent in self.sim_agents:
            self.draw_agent(agent)

        if self.play:
            pygame.display.flip()
            pygame.display.update()

     # PROJECT INVOLVED THIS FUNCTION CHANGE.
    def draw_gridsquare(self, gs):
        sl = self.scaled_location(gs.location)
        fill = pygame.Rect(sl[0], sl[1], self.scale, self.scale)

        if isinstance(gs, Counter):
            pygame.draw.rect(self.screen, Color.COUNTER, fill)
            pygame.draw.rect(self.screen, Color.COUNTER_BORDER, fill, 1)

        elif isinstance(gs, Delivery):
            pygame.draw.rect(self.screen, Color.DELIVERY, fill)
            self.draw('delivery', self.tile_size, sl)

        elif isinstance(gs, Cutboard):
            pygame.draw.rect(self.screen, Color.COUNTER, fill)
            pygame.draw.rect(self.screen, Color.COUNTER_BORDER, fill, 1)
            self.draw('cutboard', self.tile_size, sl)
        
        # Additional objects/equipments implemented in this project
        elif isinstance(gs, CookingPan):
            pygame.draw.rect(self.screen, Color.COUNTER, fill)
            pygame.draw.rect(self.screen, Color.COUNTER_BORDER, fill, 1)
            self.draw('Cookingpan', self.tile_size, sl)
        
        elif isinstance(gs, Fryer):
            pygame.draw.rect(self.screen, Color.COUNTER, fill)
            pygame.draw.rect(self.screen, Color.COUNTER_BORDER, fill, 1)
            self.draw('Fryer', self.tile_size, sl)
        
        elif isinstance(gs, PizzaOven):
            pygame.draw.rect(self.screen, Color.COUNTER, fill)
            pygame.draw.rect(self.screen, Color.COUNTER_BORDER, fill, 1)
            self.draw('PizzaOven', self.tile_size, sl)
        
        elif isinstance(gs, Sink):
            pygame.draw.rect(self.screen, Color.COUNTER, fill)
            pygame.draw.rect(self.screen, Color.COUNTER_BORDER, fill, 1)
            self.draw('Sink', self.tile_size, sl)
        
        elif isinstance(gs, TrashCan):
            pygame.draw.rect(self.screen, Color.COUNTER, fill)
            pygame.draw.rect(self.screen, Color.COUNTER_BORDER, fill, 1)
            self.draw('TrashCan', self.tile_size, sl)

        return

    def draw(self, path, size, location):
        image_path = '{}/{}.png'.format(graphics_dir, path)
        image = pygame.transform.scale(get_image(image_path), size)
        self.screen.blit(image, location)

     # PROJECT INVOLVED THIS FUNCTION CHANGE.
    def draw_agent(self, agent):
        """
        Draw agents based on their roles to assign colors.
        
        Args:
            agent: The current simulation agent
        """
        if ("Chopper" == agent.role.name) or ("BakingWaiter" == agent.role.name) or ("FryingWaiter" == agent.role.name):
            self.draw('agent-{}-{}'.format("blue", agent.role.name),
                self.tile_size, self.scaled_location(agent.location))
        elif ("Baker" == agent.role.name) or ("Deliverer" == agent.role.name) or ("ChoppingWaiter" == agent.role.name):
            self.draw('agent-{}-{}'.format("green", agent.role.name),
                self.tile_size, self.scaled_location(agent.location))
        elif ("Merger" == agent.role.name) or ("Frier" == agent.role.name) or ("Cleaner" == agent.role.name) or ("ExceptionalChef" == agent.role.name):
            self.draw('agent-{}-{}'.format("magenta", agent.role.name),
                self.tile_size, self.scaled_location(agent.location))
        else:
            self.draw('agent-{}-{}'.format("yellow", agent.role.name),
                self.tile_size, self.scaled_location(agent.location))
        self.draw_agent_object(agent.holding)

     # PROJECT INVOLVED THIS FUNCTION CHANGE.
    def draw_agent_object(self, obj):
        # Holding shows up in bottom right corner.
        if obj is None: return
        for i in obj.contents:
            if (isinstance(i, Plate)) and (i.state_index == 0):
                # Account for dirty plate action.
                self.draw('DirtyPlate', self.tile_size, self.scaled_location(obj.location))
                return
        if any([isinstance(c, Plate) for c in obj.contents]): 
            self.draw('Plate', self.holding_size, self.holding_location(obj.location))
            if len(obj.contents) > 1:
                plate = obj.unmerge('Plate')
                self.draw(obj.full_name, self.holding_container_size, self.holding_container_location(obj.location))
                obj.merge(plate)
        else:
            self.draw(obj.full_name, self.holding_size, self.holding_location(obj.location))
    
     # PROJECT INVOLVED THIS FUNCTION CHANGE.
    def get_plate_location(self):
        """
        Get a list of plate locations in the global variable.
        """
        objs = []
        for o_list in self.world.objects.values():
            for o in o_list:
                if isinstance(o, GridSquare):
                    pass
                elif o.is_held == False:
                    objs.append(o)
        
        # Draw objects not held by agents
        for o in objs:
            if any([isinstance(c, Plate) for c in o.contents]):
                Game.plate_location.append(o.location)

     # PROJECT INVOLVED THIS FUNCTION CHANGE.
    def get_all_food_plate_location(self):
        """
        Get positions of all food and plates. Utilized
        in the trashcan procedure in the interact.py for 
        dumping and return object locations.
        """
        objs = []
        alphabetClassPair = [(Fish, 'f'), (FriedChicken, 'k'), (BurgerMeat, 'm'),
                         (PizzaDough, 'P'), (Cheese, 'c'), (Bread, 'b'), (Onion, 'o'),
                         (Lettuce, 'l'), (Tomato, 't'), (Plate, 'p')]
        for o_list in self.world.objects.values():
            for o in o_list:
                if isinstance(o, GridSquare):
                    pass
                elif o.is_held == False:
                    objs.append(o)
        
        for o in objs:
            for i in range(len(o.contents)):
                for j in range(len(alphabetClassPair)):
                    if type(o.contents[i]) == alphabetClassPair[j][0]:
                        Game.food_locations.append((alphabetClassPair[j][1], o.location))
        
     # PROJECT INVOLVED THIS FUNCTION CHANGE.
    def draw_object(self, obj):
        if obj is None: return

        for i in obj.contents:
            if (isinstance(i, Plate)) and (i.state_index == 0):
                # Account for dirty plate.
                self.draw('DirtyPlate', self.tile_size, self.scaled_location(obj.location))
                return 
        
        if any([isinstance(c, Plate) for c in obj.contents]):
            self.draw('Plate', self.tile_size, self.scaled_location(obj.location))
            if len(obj.contents) > 1:
                plate = obj.unmerge('Plate')
                self.draw(obj.full_name, self.container_size, self.container_location(obj.location))
                obj.merge(plate)
        else:
            self.draw(obj.full_name, self.tile_size, self.scaled_location(obj.location))

    def scaled_location(self, loc):
        """Return top-left corner of scaled location given coordinates loc, e.g. (3, 4)"""
        return tuple(self.scale * np.asarray(loc))

    def holding_location(self, loc):
        """Return top-left corner of location where agent holding will be drawn (bottom right corner) given coordinates loc, e.g. (3, 4)"""
        scaled_loc = self.scaled_location(loc)
        return tuple((np.asarray(scaled_loc) + self.scale*(1-self.holding_scale)).astype(int))

    def container_location(self, loc):
        """Return top-left corner of location where contained (i.e. plated) object will be drawn, given coordinates loc, e.g. (3, 4)"""
        scaled_loc = self.scaled_location(loc)
        return tuple((np.asarray(scaled_loc) + self.scale*(1-self.container_scale)/2).astype(int))

    def holding_container_location(self, loc):
        """Return top-left corner of location where contained, held object will be drawn given coordinates loc, e.g. (3, 4)"""
        scaled_loc = self.scaled_location(loc)
        factor = (1-self.holding_scale) + (1-self.container_scale)/2*self.holding_scale
        return tuple((np.asarray(scaled_loc) + self.scale*factor).astype(int))


    def on_cleanup(self):
        # pygame.display.quit()
        pygame.quit()
