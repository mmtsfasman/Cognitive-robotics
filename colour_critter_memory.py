import grid
import math

mymap="""
#######
#  M  #
#     #
#  B  #
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
    dt = 0.0005
    max_speed = 20.0
    max_rotate = 10.0
    body.turn(rotation * dt * max_rotate)
    body.go_forward(speed * dt * max_speed)
    
        


#Your model might not be a nengo.Netowrk() - SPA is permitted
model = nengo.Network()
with model:

    env = grid.GridNode(world, dt=0.005)
    
    '''INPUT COLOR/ COLOR SEQUENCE'''
    color_seq = nengo.Node([0, 0, 0, 0, 0])
    targ_color = nengo.Node(0)
    
    '''BASIC MOVEMENT'''
    movement = nengo.Node(move, size_in=2)

    #Three sensors for distance to the walls
    def detect(t):
        angles = (np.linspace(-0.5, 0.5, 3) + body.dir) % world.directions
        return [body.detect(d, max_distance=4)[0] for d in angles]
    stim_radar = nengo.Node(detect)
    
    radar = nengo.Ensemble(n_neurons=100, dimensions=4, radius=4)
    nengo.Connection(stim_radar, radar[0:3])

    in_target=nengo.Node(0)
    
    #a basic movement function that just avoids walls based
    def movement_func(x):
        if x[3]:
            turn = x[3]
        else:
            turn = x[2] - x[0]
        spd = x[1] - 0.8
        return spd, turn
    
    #the movement function is only driven by information from the radar 
    nengo.Connection(radar, movement, function=movement_func)  
    
    #if you wanted to know the position in the world, this is how to do it
    #The first two dimensions are X,Y coordinates, the third is the orientation
    #(plotting XY value shows the first two dimensions)
    def position_func(t):
        return body.x / world.width * 2 - 1, 1 - body.y/world.height * 2, body.dir / world.directions
    
    position = nengo.Node(position_func)
    
    
    '''REMEMBERING THE  POSITIONS OF COLORS'''
    
    magenta = nengo.Ensemble(n_neurons=500, dimensions=2)
    nengo.Connection(magenta, magenta, synapse=.05)
    red = nengo.Ensemble(n_neurons=500, dimensions=2)
    nengo.Connection(red, red, synapse=.05)
    blue = nengo.Ensemble(n_neurons=500, dimensions=2)
    nengo.Connection(blue, blue, synapse=.05)
    green = nengo.Ensemble(n_neurons=500, dimensions=2)
    nengo.Connection(green, green, synapse=.05)
    yellow = nengo.Ensemble(n_neurons=500, dimensions=2)
    nengo.Connection(yellow, yellow, synapse=.05)

    #Ensembles estimating difference between current position and which color it is
    diff_blue = nengo.Ensemble(n_neurons=500, dimensions=4)
    diff_red = nengo.Ensemble(n_neurons=500, dimensions=4)
    diff_magenta = nengo.Ensemble(n_neurons=500, dimensions=4)
    diff_yellow = nengo.Ensemble(n_neurons=500, dimensions=4)
    diff_green = nengo.Ensemble(n_neurons=500, dimensions=4)
    
    #colors: #'G' = 1    #'R' = 2    #'B' = 3    #'M' = 4    #'Y' = 5
    
    #Sensor nodes denoting whether agent is currently in a certain color (in Boolean values)
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

    #color position memory connections 
    nengo.Connection(position[:2], diff_blue[:2])
    nengo.Connection(blue, diff_blue[:2], function=lambda x: -x, synapse=.01)
    nengo.Connection(diff_blue, blue[:2], function=lambda x: x[0:2]*x[2], synapse=.01)

    nengo.Connection(position[:2], diff_yellow[:2])
    nengo.Connection(yellow, diff_yellow[:2], function=lambda x: -x, synapse=.01)
    nengo.Connection(diff_yellow, yellow[:2], function=lambda x: x[0:2]*x[2], synapse=.01)
    
    nengo.Connection(position[:2], diff_red[:2])
    nengo.Connection(red, diff_red[:2], function=lambda x: -x, synapse=.01)
    nengo.Connection(diff_red, red, function=lambda x: x[0:2]*x[2], synapse=.01)
    
    nengo.Connection(position[:2], diff_magenta[:2])
    nengo.Connection(magenta[:2], diff_magenta[:2], function=lambda x: -x, synapse=.01)
    nengo.Connection(diff_magenta, magenta, function=lambda x: x[0:2]*x[2], synapse=.01)
    
    nengo.Connection(position[:2], diff_green[:2])
    nengo.Connection(green, diff_green[:2], function=lambda x: -x, synapse=.01)
    nengo.Connection(diff_green, green, function=lambda x: x[0:2]*x[2], synapse=.01)
    
    
    
    '''STOPPING IN THE TARGET COLOR'''
    targ_position = nengo.Ensemble(n_neurons=100, dimensions=2)
    nengo.Connection(targ_position, targ_position)
    
    def color_vector(t):
        output_vector = [float(0)]*5
        if t != 0:
            output_vector[int(t-1)] = float(1)
        print(output_vector)
        return output_vector
        
    color_processing = nengo.Ensemble(n_neurons=100, dimensions=5)
    nengo.Connection(targ_color, color_processing, function=color_vector)
    
    #assign true/false values to whether color is the target through connection with input color vector
    nengo.Connection(color_processing[1], diff_red[3])
    nengo.Connection(color_processing[0], diff_green[3])
    nengo.Connection(color_processing[2], diff_blue[3])
    nengo.Connection(color_processing[3], diff_magenta[3])
    nengo.Connection(color_processing[4], diff_yellow[3])
    
    
    nengo.Connection(diff_red, targ_position, function=lambda x: -x[0:2]*x[3], synapse=.01)
    nengo.Connection(diff_magenta, targ_position, function=lambda x:-x[0:2]*x[3], synapse=.01)
    nengo.Connection(diff_blue, targ_position, function=lambda x:-x[0:2]*x[3], synapse=.01)
    nengo.Connection(diff_green, targ_position, function=lambda x:-x[0:2]*x[3], synapse=.01)
    nengo.Connection(diff_yellow, targ_position, function=lambda x:-x[0:2]*x[3], synapse=.01)
    
    nengo.Connection(targ_position, radar[3], function=lambda x: math.atan2(x[0], x[1]))
    
    
    #basal_ganglia = nengo.networks.BasalGanglia(dimensions=2)
    
    