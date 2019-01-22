import grid

mymap="""
#######
#  M  #
# # # #
# #B# #
#G Y R#
#######
"""

class Cell(grid.Cell):

    def color(self):
        if self.wall:
            return 'black'
        elif self.cellcolor == 1:
            return 'green'
        elif self.cellcolor == 2:
            return 'red'
        elif self.cellcolor == 3:
            return 'blue'
        elif self.cellcolor == 4:
            return 'magenta'
        elif self.cellcolor == 5:
            return 'yellow'
             
        return None

    def load(self, char):
        self.cellcolor = 0
        if char == '#':
            self.wall = True
            
        if char == 'G':
            self.cellcolor = 1
        elif char == 'R':
            self.cellcolor = 2
        elif char == 'B':
            self.cellcolor = 3
        elif char == 'M':
            self.cellcolor = 4
        elif char == 'Y':
            self.cellcolor = 5
            
            
world = grid.World(Cell, map=mymap, directions=4)

body = grid.ContinuousAgent()
world.add(body, x=1, y=2, dir=2)

import nengo
import numpy as np 
'''import nengo.spa as spa
from nengo.spa import Vocabulary

D = 32  # the dimensionality of the vectors
rng = np.random.RandomState(7)
vocab = Vocabulary(dimensions=D, rng=rng, max_similarity=0.1)

#Adding semantic pointers to the vocabulary
RED=vocab.parse('RED')
BLUE=vocab.parse('BLUE')
MAGENTA=vocab.parse('MAGENTA')
YELLOW=vocab.parse('YELLOW')
ZERO=vocab.add('ZERO', [0]*D)

model = spa.SPA(label="Question Answering with Memory", vocabs=[vocab])
'''

def move(t, x):
    speed, rotation = x
    dt = 0.001
    max_speed = 20.0
    max_rotate = 10.0
    body.turn(rotation * dt * max_rotate)
    body.go_forward(speed * dt * max_speed)
    
    
def move_to_col(t, x):
    
    col =  x[0]
    targ_col = x[1]
    speed = x[2]
    rotation = x[3]
    
    if col != targ_col:
        dt = 0.001
        max_speed = 20.0
        max_rotate = 10.0
        body.turn(rotation * dt * max_rotate)
        body.go_forward(speed * dt * max_speed)
    else:
        speed=0
        rotation = 0
        dt = 0.001
        max_speed = 20.0
        max_rotate = 10.0
        body.turn(0)
        body.go_forward(0)
        


#Your model might not be a nengo.Netowrk() - SPA is permitted
model = nengo.Network()
with model:
    '''
    model.col1 = spa.State(D, label='color 1')
    model.col2 = spa.State(D, label='color 2')
    model.col3 = spa.State(D, label='color 3')
    model.col4 = spa.State(D, label='color 4')
    model.col5 = spa.State(D, label='color 5')
    model.bound = spa.State(D, label='bound')
    model.memory = spa.State(D, label='memory')
    
    model.input = spa.Input(col1=color_input, B=shape_input, C=cue_input)   
    '''
    
    
    magenta = nengo.Ensemble(n_neurons=500, dimensions=2)
    nengo.Connection(magenta, magenta, synapse=0.01)
    red = nengo.Ensemble(n_neurons=500, dimensions=2)
    nengo.Connection(red, red, synapse=0.01)
    blue = nengo.Ensemble(n_neurons=500, dimensions=2)
    nengo.Connection(blue, blue, synapse=0.01)
    green = nengo.Ensemble(n_neurons=500, dimensions=2)
    nengo.Connection(green, green, synapse=0.01)
    yellow = nengo.Ensemble(n_neurons=500, dimensions=2)
    nengo.Connection(yellow, yellow, synapse=0.01)
    
    
    env = grid.GridNode(world, dt=0.005)
    

    movement = nengo.Node(move, size_in=2)
    #movement = nengo.Node(move_to_col, size_in=4)
    

    #Three sensors for distance to the walls
    def detect(t):
        angles = (np.linspace(-0.5, 0.5, 3) + body.dir) % world.directions
        return [body.detect(d, max_distance=4)[0] for d in angles]
    stim_radar = nengo.Node(detect)
    
    
    #This node returns the colour of the cell currently occupied. Note that you might want to transform this into 
    #something else (see the assignment)
    
    #def map_colors(color):
        
        
        
    #color_map = nengo.Ensemble(n_neurons=100, dimensions=4)
    
    radar = nengo.Ensemble(n_neurons=100, dimensions=3, radius=4)
    nengo.Connection(stim_radar, radar)
    #nengo.Connection(current_color, radar[0])
    #nengo.Connection(targ_color, radar[1])
    
    def sensor(t, x):
        if x[0] == x[1]:
            return 1
        else:
            return 0
    
    
            
    #nengo.Connection(radar[0:1], color_diff, function=color_differ)
    
    #a basic movement function that just avoids walls based
    def movement_func(x):
        turn = x[2] - x[0]
        spd = x[1] - 0.8
        #print(x[1])
        #move_to_col(turn, spd, x[3], x[4])
        return spd, turn
    
    #the movement function is only driven by information from the
    #radar 
    nengo.Connection(radar, movement, function=movement_func)  
    #nengo.Connection(targ_color, movement[4]) 
    
    
    #if you wanted to know the position in the world, this is how to do it
    #The first two dimensions are X,Y coordinates, the third is the orientation
    #(plotting XY value shows the first two dimensions)
    def position_func(t):
        return body.x / world.width * 2 - 1, 1 - body.y/world.height * 2, body.dir / world.directions
    
    position = nengo.Node(position_func)
    
    
    def position_func(t):
        print(body.cell.cellcolor)
        return body.x / world.width * 2 - 1, 1 - body.y/world.height * 2, body.dir / world.directions
    
    diff_blue = nengo.Ensemble(500, 3)
    diff_red = nengo.Ensemble(500, 3)
    diff_magenta = nengo.Ensemble(500, 3)
    diff_yellow = nengo.Ensemble(500, 3)
    diff_green = nengo.Ensemble(500, 3)
    
    #'G' = 1
    #'R' = 2
    #'B' = 3
    #'M' = 4
    #'Y' = 5
    
    
    in_blue =       nengo.Node(output=lambda t: int(body.cell.cellcolor==3))
    in_yellow =     nengo.Node(output=lambda t: int(body.cell.cellcolor==5))
    in_red =        nengo.Node(output=lambda t: int(body.cell.cellcolor==2))
    in_magenta =    nengo.Node(output=lambda t: int(body.cell.cellcolor==4))
    in_green =      nengo.Node(output=lambda t: int(body.cell.cellcolor==1))
    
    nengo.Connection(in_blue, diff_blue[2])
    nengo.Connection(in_yellow, diff_yellow[2])
    nengo.Connection(in_red, diff_red[2])
    nengo.Connection(in_magenta, diff_magenta[2])
    nengo.Connection(in_green, diff_green[2])

    
    nengo.Connection(position[:2], diff_blue[:2])
    nengo.Connection(blue, diff_blue[:2], function=lambda x: -x, synapse=0.001)
    nengo.Connection(diff_blue, blue, function=lambda x: x[0:2]*x[2], synapse=.001)

    nengo.Connection(position[:2], diff_yellow[:2])
    nengo.Connection(yellow, diff_yellow[:2], function=lambda x: -x, synapse=0.001)
    nengo.Connection(diff_yellow, yellow, function=lambda x: x[0:2]*x[2], synapse=.001)
    
    nengo.Connection(position[:2], diff_red[:2])
    nengo.Connection(red, diff_red[:2], function=lambda x: -x, synapse=0.001)
    nengo.Connection(diff_red,red, function=lambda x: x[0:2]*x[2], synapse=.001)
    
    nengo.Connection(position[:2], diff_magenta[:2])
    nengo.Connection(magenta, diff_magenta[:2], function=lambda x: -x, synapse=0.001)
    nengo.Connection(diff_magenta, magenta, function=lambda x: x[0:2]*x[2], synapse=.001)
    
    nengo.Connection(position[:2], diff_green[:2])
    nengo.Connection(green, diff_green[:2], function=lambda x: -x, synapse=0.001)
    nengo.Connection(diff_green, green, function=lambda x: x[0:2]*x[2], synapse=.001)
    