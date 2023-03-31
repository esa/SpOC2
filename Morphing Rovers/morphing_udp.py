# Morphing Rover Challenge
# GECCO 2023 Space Optimisation Competition (SPoC)

### LOADING PACKAGES
############################################################################################################################################

import numpy as np
from math import atan2
from math import floor
from math import exp
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import rotate, gaussian_blur
from collections import defaultdict
import matplotlib.pyplot as plt
import os

### CONSTANTS DEFINING THE PROBLEM
############################################################################################################################################

# File path to the data required for this challenge
# If the files are in the folder PATH = './myfolder', then the structure has to be:
# maps: './myfolder/Maps'
# coordinates: './myfolder/coordinates.txt'
# example chromosome: './myfolder/example_rover.npy'
PATH = os.path.join(".", "data", "spoc2", "morphing")

# Parameters for the rover modes
MASK_SIZE = 11
NUMBER_OF_MODES = 4
NUM_MODE_PARAMETERS = NUMBER_OF_MODES*MASK_SIZE**2
MASK_CENTRES = []
for m_id in range(NUMBER_OF_MODES):
    MASK_CENTRES.append(int(m_id*MASK_SIZE**2+0.5*MASK_SIZE**2))
    
# Size and field of view of rover
FIELD_OF_VIEW = int(MASK_SIZE/2+1)
VISIBLE_SIZE = int(8*MASK_SIZE)
MIN_BORDER_DISTANCE = int(0.6*VISIBLE_SIZE)

# Cooldown of morphing
MODE_COOLDOWN = int(VISIBLE_SIZE/MASK_SIZE)

# Minimum distance when sample is counted as collected
SAMPLE_RADIUS = FIELD_OF_VIEW

# Parameters of the neural network controlling the rover
NETWORK_SETUP = {'filters': 8, 
                 'kernel_size': MASK_SIZE, 
                 'stride': 2, 
                 'dilation': 1,
                 'filters1': 16, 
                 'kernel_size1': 4,
                 'pooling_size': 2,
                 'state_neurons': 40, 
                 'hidden_neurons': [40,40]}

# Rover dynamics
DELTA_TIME = 1
MAX_TIME = 500
SIM_TIME_STEPS = int(MAX_TIME/DELTA_TIME)
MAX_VELOCITY = MASK_SIZE
MAX_DV = DELTA_TIME * MAX_VELOCITY
MAX_ANGULAR_VELOCITY = np.pi/4.
MAX_DA = DELTA_TIME * MAX_ANGULAR_VELOCITY

# Number of maps and scenarios per map
TOTAL_NUM_MAPS = 6
MAPS_PER_EVALUATION = 6
SCENARIOS_PER_MAP = 5
TOTAL_NUM_SCENARIOS = MAPS_PER_EVALUATION*SCENARIOS_PER_MAP

# File path and names
HEIGHTMAP_NAMES = ['Map1.jpg', 'Map2.JPG', 'Map3.JPG', 'Map4.JPG', 'Map5.JPG', 'Map6.jpg']
COORDINATE_NAME = '{}/coordinates.txt'.format(PATH)
# Kernel size for smoothing maps a little bit with a Gaussian kernel
BLUR_SIZE = 7

# Loading raw data
COORDINATE_FILE = open(COORDINATE_NAME, 'r')
COORDINATES = [[] for map in range(TOTAL_NUM_MAPS)]
for entry in COORDINATE_FILE.readlines():
    entry = entry.split('\t')
    COORDINATES[int(entry[0])].append([float(x) for x in entry[1:5]])
SCENARIO_POSITIONS = torch.Tensor(COORDINATES)

# Constants used for numerical stability and parameter ranges
EPS_C = (0.03)**2
FLOAT_MIN = -100
FLOAT_MAX = 100
CENTRE_MIN = 1e-16

# Initialising constants for extracting the map terrain the rover is on
VIEW_LEFT = int(VISIBLE_SIZE/2)
VIEW_RIGHT = VIEW_LEFT+1
MODE_VIEW_LEFT = int(MASK_SIZE/2)
MODE_VIEW_RIGHT = MODE_VIEW_LEFT+1

### UTILITY FUNCTIONS
############################################################################################################################################
    
def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    '''
    Utility function for computing output of convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    
    From https://discuss.pytorch.org/t/utility-function-for-calculating-the-shape-of-a-conv-output/11173/5
    '''
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    h = floor( ((h_w[0] + (2 * pad) - ( dilation * (kernel_size[0] - 1) ) - 1 )/ stride) + 1)
    w = floor( ((h_w[1] + (2 * pad) - ( dilation * (kernel_size[1] - 1) ) - 1 )/ stride) + 1)
    return h, w

def get_conv_size(network_setup):
    '''
    Function returning the layer size after two convolutions in a neural network.
    '''
    cwidth,cheight = conv_output_shape([VISIBLE_SIZE+1,VISIBLE_SIZE+1],
                                       network_setup['kernel_size'],
                                       network_setup['stride'],
                                       0,
                                       network_setup['dilation'])
    cwidth,cheight = conv_output_shape([cwidth,cheight],
                                       network_setup['pooling_size'],
                                       network_setup['pooling_size'],
                                       0,
                                       1)
    cwidth,cheight = conv_output_shape([cwidth,cheight],
                                        network_setup['kernel_size1'],
                                       network_setup['stride'],
                                       0,
                                       network_setup['dilation'])
    cwidth,cheight = conv_output_shape([cwidth,cheight],
                                       network_setup['pooling_size'],
                                       network_setup['pooling_size'],
                                       0,
                                       1)
    conv_size = cwidth*cheight*network_setup['filters1']
    return conv_size

def get_number_of_parameters(network_setup):
    '''
    Function returning the number of biases, weights and size of the convolutional layer given a neural network setup.
    '''
    number_biases = 2+network_setup['filters'] + network_setup['filters1'] + network_setup['state_neurons'] + network_setup['hidden_neurons'][0] + network_setup['hidden_neurons'][1]
    
    conv_size = get_conv_size(network_setup)
    
    number_weights = conv_size*network_setup['hidden_neurons'][0]+\
                  network_setup['state_neurons']*network_setup['hidden_neurons'][0]+\
                  network_setup['hidden_neurons'][0]*network_setup['hidden_neurons'][1]+\
                  network_setup['hidden_neurons'][1]*2 + (NUMBER_OF_MODES+5)*network_setup['state_neurons']+network_setup['hidden_neurons'][1]**2+\
                  network_setup['filters']*network_setup['kernel_size']**2 +network_setup['filters']*network_setup['filters1']*network_setup['kernel_size1']**2
    
    return number_biases, number_weights, conv_size

def minimal_angle_diff(target_angle, moving_angle):
    '''
    Calculates the smallest absolute angle difference between two angles.
    Angles are measured w.r.t. the x-axis (i.e., pointing eastward), and are 
    increased counter-clockwise when rotating the vector in the x-y plane.

    Angles are given in [rad] here.
    
    Args:
        target_angle: Angle of vector from origin to target.
        moving_angle: Angle of vector from origin to rover.
    
    Returns:
        min_angle: positive or negative angular difference required to 
                   overlap target_angle and moving_angle.
    '''
    angle_diff = target_angle - moving_angle
    min_angle = ((angle_diff+np.pi)%(2*np.pi))-np.pi
    return min_angle

def velocity_function(form, mode_view):
    '''
    Velocity function that maps a rover form and current terrain to a velocity.

    Composed of a term taking into account shape only (distance)
    as well as height difference scale only (luminance).

    Args:
        form: mask of the rover mode
        mode_view: terrain height map the rover is standing on.

    Returns:
        Scalar between 0 and 1 that scales the velocity.
        Rover velocity is obtained by multiplying this factor with MAX_VELOCITY.
    '''
    # calculate norm of vectors
    f_norm = form.norm()
    mv_norm = mode_view.norm()
    # luminance = how well do the vector norms agree?
    # 0 = no match, 1 = perfect match
    luminance = (2*f_norm*mv_norm+EPS_C)/(f_norm**2+mv_norm**2+EPS_C)
    # distance = Euclidean distance on unit sphere (i.e., of normalized vectors)
    # Rescaled to go from 0 to 1, with 0 = no match, 1 = perfect match
    distance = (2-(form/f_norm - mode_view/mv_norm).norm())*0.5
    # final metric = product of distance and luminance
    metric = distance*luminance.sqrt()
    # turn metric into range 0 to 2
    # 0 = perfect match, 2 = worst match
    metric = (1-metric)*2
    return distance_to_velocity(metric)

def distance_to_velocity(x):
    '''
    Helper function that turns the initial score for rover and terrain similarity into a velocity.
    '''
    return 1./(1+x**3)

class Record:
    def __init__(self):
        '''
        Convenience class for recording data from simulations.
        '''
        self.data = [[defaultdict(list) for i in range(SCENARIOS_PER_MAP)] for j in range(MAPS_PER_EVALUATION)]

    def __getitem__(self, item):
        '''
        Access data with bracket notation.

        E.g., 
        recorder = Record()
        recorder.add(0,0, {...})
        print(recorder[0][0])
        '''
        return self.data[item]
    
    def add(self, map_id, scenario_id, variables):
        '''
        Append recorded data for a map and scenario.

        Args:
            map_id: ID of the map data was recorded on.
            scenario_id: ID of the scenario.
            variables: dictionary with key - value pairs for different variables that have been recorded.
                       Value can be a torch tensor, numpy array, list or scalar.
        '''
        for key in variables:
            if isinstance(variables[key], torch.Tensor):
                value = variables[key].detach().numpy().tolist()
            elif isinstance(variables[key], np.ndarray):
                value = variables[key].tolist()
            else:
                value = variables[key]
            self.data[map_id][scenario_id][key].append(value)
            
def ax_for_plotting(ax, map_id, scenario_id):
    '''
    Convenience function for plotting.

    Args:
        ax: matplotlib ax object to plot in.
        map_id: ID of the map.
        scenario_id: ID of the scenario.
    '''
    if MAPS_PER_EVALUATION == 1 and SCENARIOS_PER_MAP == 1:
        return ax
    elif MAPS_PER_EVALUATION == 1:
        return ax[scenario_id]
    elif SCENARIOS_PER_MAP == 1:
        return ax[map_id]
    else:
        return ax[map_id][scenario_id]
                

### CONSTANTS FOR NEURAL NETWORK
############################################################################################################################################

NUM_BIASES, NUM_WEIGHTS, CONV_SIZE = get_number_of_parameters(NETWORK_SETUP)
NUM_NN_PARAMS = NUM_BIASES + NUM_WEIGHTS

### NEURAL NETWORK CONTROLLING THE ROVER
############################################################################################################################################

class Controller(nn.Module):
    def __init__(self, chromosome):
        '''
        Neural network that controls the rover.

        Initialized from a chromosome specifying the biases, weights and the type of
        pooling layers and activation functions per layer.

        By default, gradient calculation is turned off. If required, please remove

        'self._turn_off_gradients()'

        in the init method.
        '''
        super().__init__()
            
        # Split up chromosome
        bias_chromosome = chromosome[:NUM_BIASES]
        weight_chromosome = chromosome[NUM_BIASES:NUM_NN_PARAMS]
        self.network_chromosome = chromosome[NUM_NN_PARAMS:]
        
        # Decode network chromosome
        pooling1 = int(self.network_chromosome[0])
        pooling2 = int(self.network_chromosome[1])
        atype1 = int(self.network_chromosome[2])
        atype2 = int(self.network_chromosome[3])
        atype3 = int(self.network_chromosome[4])
        atype4 = int(self.network_chromosome[5])
        atype5 = int(self.network_chromosome[6])

        # Set up chosen pooling operator and activation functions
        self.pool1 = self._init_pooling_layer(pooling1)
        self.pool2 = self._init_pooling_layer(pooling2)

        self.activation1 = self._init_activation_function(atype1)
        self.activation2 = self._init_activation_function(atype2)
        self.activation3 = self._init_activation_function(atype3)
        self.activation4 = self._init_activation_function(atype4)
        self.activation5 = self._init_activation_function(atype5)
        
        # Input 1: used rover mode (one-hot), angle to target, distance to target
        # Input 2: map of local landscape
        self.inp = nn.Linear(NUMBER_OF_MODES+5, NETWORK_SETUP['state_neurons'])
        self.conv = nn.Conv2d(in_channels=1, 
                              out_channels=NETWORK_SETUP['filters'], 
                              kernel_size=NETWORK_SETUP['kernel_size'],
                              stride=NETWORK_SETUP['stride'],
                              dilation=NETWORK_SETUP['dilation'])
        self.conv2 = nn.Conv2d(in_channels=NETWORK_SETUP['filters'], 
                              out_channels=NETWORK_SETUP['filters1'], 
                              kernel_size=NETWORK_SETUP['kernel_size1'],
                              stride=NETWORK_SETUP['stride'],
                              dilation=NETWORK_SETUP['dilation'])
        
        # Remaining network
        self.lin2 = nn.Linear(CONV_SIZE, NETWORK_SETUP['hidden_neurons'][0], bias = False)
        self.lin3 = nn.Linear(NETWORK_SETUP['state_neurons'], NETWORK_SETUP['hidden_neurons'][0])

        self.lin4 = nn.Linear(NETWORK_SETUP['hidden_neurons'][0], NETWORK_SETUP['hidden_neurons'][1])
        self.recurr = nn.Linear(NETWORK_SETUP['hidden_neurons'][1], NETWORK_SETUP['hidden_neurons'][1], bias = False)
        self.output = nn.Linear(NETWORK_SETUP['hidden_neurons'][1], 2)

        self._turn_off_gradients()

        # Load weights and biases from chromosomes
        self._set_weights_from_chromosome(weight_chromosome)
        self._set_biases_from_chromosome(bias_chromosome)
    
    def forward(self, landscape, state, past_inp):
        '''
        Given the surrounding landscape, rover state and previous network state, 
        return: 
            - mode control (whether to switch mode) 
            - angle control (how to change the orientation of the rover).
            - latent activity of the neural network to be passed to the next iteration.
        '''
        # Add batch and channel dimension if necessary
        if len(landscape.size()) == 2:
            landscape = landscape.unsqueeze(0)
        if len(landscape.size()) == 3:
            landscape = landscape.unsqueeze(0)
        if len(state.size()) == 1:
            state = state.unsqueeze(0)
        if len(past_inp.size()) == 1:
            past_inp = past_inp.unsqueeze(0)
        
        # Forward propagation
        # Separate pathways for modalities
        x, y = self.conv(landscape), self.inp(state)
        x, y = self.activation1(x), self.activation2(y)
        x = self.pool1(x)
        x = self.activation3(self.conv2(x))
        x = self.pool2(x).flatten(1)
        # Combine information in common hidden layer
        x = self.lin2(x) + self.lin3(y)
        x = self.activation4(x)
        # Apply another hidden layer + recurrence (memory)
        x = self.lin4(x) + self.recurr(past_inp)
        xlat = self.activation5(x)
        # Get output
        x = self.output(xlat)
        # Rover mode switch command
        mode_command = x[:,0]
        # Rover orientation
        angle_command = torch.clamp(x[:,-1], min = -1, max = 1)
        
        return mode_command, angle_command, xlat
    
    @property
    def chromosome(self):
        '''
        Return chromosome that defines the whole network.
        '''
        chromosomes = {'weights': torch.Tensor([]), 'biases': torch.Tensor([])}
        for param in self.parameters():
            shape = list(param.size())
            if len(shape)>1:
                whichone = 'weights'
            else:
                whichone = 'biases'
            chromosomes[whichone] = torch.concat([chromosomes[whichone], param.flatten().detach()])    
        
        final_chromosome = list(chromosomes['biases'].detach().numpy())+\
                           list(chromosomes['weights'].detach().numpy())+\
                           list(self.network_chromosome)
        
        return final_chromosome
    
    def _set_weights_from_chromosome(self, chromosome):
        '''
        Set the weights from a flat vector.
        '''
        if not isinstance(chromosome, torch.Tensor):
            chromosome = torch.Tensor(chromosome)
        prev_slice, next_slice = 0, 0
        for param in self.parameters():
            shape = list(param.size())
            if len(shape)>1:
                next_slice += np.prod(shape)
                slices = chromosome[prev_slice:next_slice]
                param.data = slices.reshape(shape)
                prev_slice = next_slice
        assert(prev_slice == NUM_WEIGHTS)
    
    def _set_biases_from_chromosome(self, chromosome):
        '''
        Set the biases from a flat vector.
        '''
        if not isinstance(chromosome, torch.Tensor):
            chromosome = torch.Tensor(chromosome)
        prev_slice, next_slice = 0, 0
        for param in self.parameters():
            shape = list(param.size())
            if len(shape)==1:
                next_slice += shape[0]
                param.data = chromosome[prev_slice:next_slice]
                prev_slice = next_slice
        assert(prev_slice == NUM_BIASES)

    def _init_pooling_layer(self, chromosome):
        '''
        Convenience function for setting the pooling layer.
        '''
        size = NETWORK_SETUP['pooling_size']
        if chromosome == 0:
            return nn.MaxPool2d(size)
        elif chromosome == 1:
            return nn.AvgPool2d(size)
        else:
            raise Exception('Pooling type with ID {} not implemented.'.format(chromosome))
            
    def _init_activation_function(self, chromosome):
        '''
        Convenience function for setting the activation function.
        '''
        if chromosome == 0:
            return nn.Sigmoid()
        elif chromosome == 1:
            return nn.Hardsigmoid()
        elif chromosome == 2:
            return torch.tanh
        elif chromosome == 3:
            return nn.Hardtanh()
        elif chromosome == 4:
            return nn.Softsign()
        elif chromosome == 5:
            return nn.Softplus()
        elif chromosome == 6:
            return F.relu
        else:
            raise Exception('Activation type with ID {} not implemented.'.format(chromosome))

    def _turn_off_gradients(self):
        '''
        Convenience function that turns off gradient calculation for all network parameters.
        '''
        for param in self.parameters():
            param.requires_grad = False

### ROVER CLASS
############################################################################################################################################

class Rover:
    def __init__(self, chromosome):
        '''
        Class defining the rover. Contains both the rover forms, the control neural network as well as 
        a function for updating the state of the rover given its surroundings and current state.
        '''
        # Extract forms from chromosome
        form_chromosome = chromosome[:NUM_MODE_PARAMETERS]
        # Extract neural network parameters from chromosome
        network_chromosome = chromosome[NUM_MODE_PARAMETERS:]
        # Create control neural network
        self.Control = Controller(network_chromosome)
        # Create form masks for the rover modes
        self.form_masks = torch.Tensor(np.reshape(form_chromosome, (NUMBER_OF_MODES, MASK_SIZE, MASK_SIZE)))
        
        # Initialize the rover state
        self.current_mode = 0 # current mode, digit from 0 to NUM_ROVER_MODES-1
        self.cooldown = 0 # cooldown of switching between modes
        self.mode_efficiency = 0 # current efficiency of the rover mode, only used for recording
        self.position = None # position of the rover
        self.angle = 0 # orientation of the rover w.r.t. x-axis
        self.latent_state = None

    @property
    def chromosome(self):
        '''
        Return the chromosome defining this rover.
        '''
        return list(self.form_masks.flatten().detach().numpy())+self.Control.chromosome
        
    @property
    def direction(self):
        '''
        Return unit vector pointing in the direction the rover is looking.
        '''
        return torch.Tensor([np.cos(self.angle), np.sin(self.angle)])

    @property
    def onehot_representation_of_mode(self):
        '''
        Return one-hot vector representation of the active mode.

        E.g., for 4 modes, turns 0 -> [1,0,0,0], 1 -> [0,1,0,0], etc.
        '''
        return list(np.eye(1,NUMBER_OF_MODES,self.current_mode)[0])

    def reset(self, start_position):
        """
        Convenience function to reset the rover state between scenarios.
        """
        self.current_mode = 0
        self.angle = 0
        self.mode_efficiency = 0
        self.cooldown = 0
        self.position = start_position
        self.latent_state = torch.zeros(NETWORK_SETUP['hidden_neurons'][1])
    
    def velocity_calculation(self, mode_view):
        '''
        This function calculates the velocity of the rover based on the current local terrain mask and rover form mask.

        Args:
            mode_view: rotated local terrain height the rover is standing on
        Returns:
            mode_efficiency: the current normed velocity (multiples of max. velocity) of the rover
        '''
        # get mask of current mode
        form = self.form_masks[self.current_mode]
        # determine velocity factor for current form and terrain
        mode_efficiency = velocity_function(form, mode_view)
        
        return mode_efficiency
    
    def get_best_mode(self, mode_view):
        '''
        Returns the mode that yields the highest velocity on a given part of the terrain.
        
        Args:
            mode_view: rotated local terrain height the rover is standing on
        Returns:
            best_mode: best mode for the terrain the rover is standing on
        '''
        velocities = []
        for i in range(NUMBER_OF_MODES):
            form = self.form_masks[i]
            velocities.append(velocity_function(form, mode_view))
        best_mode = np.argmax(velocities)

        return best_mode
        
    def update_rover_state(self, rover_view, mode_view, distance_vector, original_distance):
        """
        Updates the rover state variables for the current timestep.

        Args:
            rover_view: the view (top-down, unrotated) of the local terrain
            mode_view: the terrain the rover is standing on
            distance_vector: the vector from the rover to the target
            original_distance: the scalar distance from the starting point to the target
        """
        # Calculate angle and distance between rover and sample
        angle_to_sample = atan2(distance_vector[1], distance_vector[0])
        distance_to_sample = distance_vector.norm()/original_distance

        # Turn angle to minimal angle (from -pi to pi)
        angle_diff =  minimal_angle_diff(angle_to_sample, self.angle)
        # Prepare state of the rover as input to the neural network
        # Contains:
        # - the efficiency of the current mode
        # - normed cooldown length (between 0 and 1)
        # - how well the orientation of the rover aligns with the target sample location
        # - relative distance to target sample location
        # - rover orientation
        # - active rover mode (one-hot coded)
        rover_state = torch.Tensor([self.mode_efficiency, self.cooldown/MODE_COOLDOWN, angle_diff/np.pi,\
                                    float(distance_to_sample), self.angle/np.pi/2]+self.onehot_representation_of_mode)
        # The neural network takes the rover view and rover state as input
        # and returns whether the rover should morph and how the orientation should be changed
        switching_mode, angular_change, self.latent_state = self.Control(rover_view, rover_state, self.latent_state)
        
        # Save angular velocity change obtained from neural network
        angular_velocity_factor = angular_change.detach().numpy()[0]
        # Calculate efficiency of current rover mode on current terrain
        # (Takes as input the terrain the rover is standing on)
        velocity_factor = self.velocity_calculation(mode_view)
        # Store mode efficiency for later visualisation
        self.mode_efficiency = velocity_factor
        
        # Logic for switching the mode
        # If morphing is not on cooldown and the neural network initiates a morph
        if (self.cooldown == 0) and (switching_mode > 0):
            # then get the currently best mode
            new_mode = self.get_best_mode(mode_view)
            # if this mode is different from the current mode (i.e., the rover has to morph),
            # set morphing on cooldown
            if new_mode != self.current_mode:
                self.cooldown = MODE_COOLDOWN+1
            # update rover mode
            self.current_mode = new_mode
        # clock down the cooldown of rover morphing
        if self.cooldown > 0:
            self.cooldown -= 1
        
        # update position and orientation of the rover
        self.position.data = self.position + MAX_DV * velocity_factor * self.direction
        self.angle = self.angle + MAX_DA * angular_velocity_factor

### CLASS HOLDING THE LANDSCAPE DATA
############################################################################################################################################

class MysteriousMars():
    def __init__(self):
        """
        This class holds the map data and a convenience function for extracting terrain views
        of the rover given its position and orientation.
        """
        # Setting up the scenario maps and positions
        self.heightmaps = [0] * MAPS_PER_EVALUATION
        self.heightmap_sizes = [0] * MAPS_PER_EVALUATION
        for counter in range(MAPS_PER_EVALUATION):
            # load maps
            self.heightmaps[counter] = torch.Tensor(imageio.imread('{}/Maps/{}'.format(PATH, HEIGHTMAP_NAMES[counter])))
            # unify array format of all maps
            if len(self.heightmaps[counter].size()) == 3:
                self.heightmaps[counter] = self.heightmaps[counter][:,:,0]
            # smooth maps using Gaussian kernel
            self.heightmaps[counter] = gaussian_blur(self.heightmaps[counter].unsqueeze(0), BLUR_SIZE).squeeze(0)
            # map sizes used to identify when the rover leaves the map
            self.heightmap_sizes[counter] = list(self.heightmaps[counter].shape)

    def extract_local_view(self, position, direction_angle, map_id):
        """
        This function extracts the local terrain view in the direction the rover is facing.

        Args:
            position: the vector position of the rover
            direction: the unit vector direction the rover is travelling in
            map_id: the ID of the height map used in the scenario
        Returns:
            rover view: top-down, unrotated local terrain view
            mode view: the terrain the rover is standing on (rotated such that upwards is equivalent 
                                                             to the direction the rover is looking at)
        """
        # Get rover position in map coordinates
        col, row = np.round(position[0]), np.round(position[1])
        col, row = int(col), self.heightmap_sizes[map_id][0]-int(row)
        
        # Get map
        height_map = self.heightmaps[map_id]
        
        # Current angle the rover is looking at
        angle = 90-direction_angle/np.pi*180

        # Slice the current view of the rover
        hmap_slice = height_map[row-VIEW_LEFT:row+VIEW_RIGHT,\
                                col-VIEW_LEFT:col+VIEW_RIGHT]
        
        # Correctly rotate the terrain the rover is standing on
        rotated_slice = rotate(hmap_slice.unsqueeze(0), angle, InterpolationMode.BILINEAR).squeeze(0)
        
        # Both rover view and mode view are normalised by subtracting the central terrain height
        rover_view = (hmap_slice - hmap_slice[VIEW_LEFT,VIEW_LEFT])/255.
        mode_view = rotated_slice[VIEW_LEFT-MODE_VIEW_LEFT:VIEW_LEFT+MODE_VIEW_RIGHT,\
                               VIEW_LEFT-MODE_VIEW_LEFT:VIEW_LEFT+MODE_VIEW_RIGHT]
        mode_view = mode_view - mode_view[MODE_VIEW_LEFT,MODE_VIEW_LEFT] + 1
        
        return rover_view, mode_view

############################################################################################################################################
##### UDP CLASS FORMULATING THE OPTIMIZATION PROBLEM
############################################################################################################################################

class morphing_rover_UDP:
    def __init__(self):
        """
        A Pygmo compatible UDP User Defined Problem representing the Morphing Rover challenge for SpOC 2023.
        https://esa.github.io/pygmo2/tutorials/coding_udp_simple.html explains what more details on UDPs.

        The morphing rover properties and the neural net architecture of the rover are defined by the chromosome/decision vector A. 
        The rover defined by A must complete a series of routes across different terrains as quickly as possible using the same forms 
        and controller each time.
        """
        # Create the planet!
        self.env = MysteriousMars()
        
    def get_bounds(self):
        """
        Get bounds for the decision variables.

        Returns:
            Tuple of lists: bounds for the decision variables.
        """
        lb, rb = [], []
        lb += [FLOAT_MIN]*NUM_MODE_PARAMETERS
        for c_id in MASK_CENTRES:
            lb[c_id] = CENTRE_MIN
        rb += [FLOAT_MAX]*NUM_MODE_PARAMETERS
        lb += [FLOAT_MIN]*(NUM_NN_PARAMS)
        rb += [FLOAT_MAX]*(NUM_NN_PARAMS)
        lb += [0,0,0,0,0,0,0]
        rb += [1,1,6,6,6,6,4]
        
        return (lb, rb)

    def get_nix(self):
        """
        Get number of integer variables in the chromosome/decision vector.

        Returns:
            int: number of integer variables.
        """
        return 7

    def fitness(self, chromosome, detailed_results=None):
        """
        Fitness function for the UDP
        
        Args:
            chromosome: the chromosome/decision vector to be tested
            detailed_results: whether to record all the results from a scenario
            pretty: if the pretty function is called, this returns scores for each map
        Returns:
            score: the score/fitness for this chromosome. Best value is 1.
        """
        # Create rover from chromosome
        rover = Rover(chromosome)

        # Initialize score
        score = 0

        # Simulates N scenarios, records the results
        for heightmap in range(MAPS_PER_EVALUATION):
            for scenario in range(SCENARIOS_PER_MAP):
                result = self.run_single_scenario(rover, heightmap, scenario, detailed_results)
                ind_score = (1+result[0]) * result[1]
                score += ind_score
                if detailed_results is not None:
                    detailed_results.add(heightmap, scenario, {'fitness': ind_score})
        score = float(score/TOTAL_NUM_SCENARIOS)
                
        return [score]
    
    def pretty(self, chromosome, verbose = True):
        '''
        Returns both the fitness and a detailed recording for each scenario,
        including rover trajectories, mode efficiencies, used modes and individual fitnesses.
        Also prints a table of fitness values for each map and scenario if verbose is True.

        Args:
            chromosome: Chromosome to be tested.
            verbose: Whether to print fitness values.

        Returns:
            score: Average fitness over all maps and scenarios
            detailed_results: recording for each map and scenario, including rover trajectories etc.
        '''
        detailed_results = Record()
        score = self.fitness(chromosome, detailed_results)
        
        # Print fitness for all scenarios in terminal
        if verbose == True:
            scores = 'Fitness\n'
            scores += '~~~~~~~\n'
            scores += '\t     '
            for i in range(SCENARIOS_PER_MAP):
                scores += 'Scenario {}\t     '.format(i+1)
            scores += '\n'
            for j in range(MAPS_PER_EVALUATION):
                for i in range(SCENARIOS_PER_MAP):
                    if i == 0:
                        scores += 'Map {}\t |\t'.format(j+1)
                    scores += '{:.3f}\t | \t'.format(detailed_results[j][i]['fitness'][0])
                scores += '\n'
            scores += '\n'
            scores += 'Avg. fitness: {:.3f}'.format(score[0])
            print(scores)
        
        return score, detailed_results
    
    def example(self):
        '''
        Load an example chromosome.
        '''
        example_chromosome = np.load('{}/example_rover.npy'.format(PATH))
        return example_chromosome

    def run_single_scenario(self, rover, map_number, scenario_number, detailed_results = None):
        """
        Function for running a single scenario on a map given a rover.
        Maximum simulation time is 500 timesteps, in which the neural network of the rover determines whether to switch
        the rover's form and how to change the orientation of the rover at each timestep. Position updates are done
        by comparing the terrain the rover is standing on with the mask of the active mode (using a function).
        If the rover reaches a certain radius around the sample the scenario is completed early. 
        If the for loop ends or the rover travels outside of the map, the function records no samples saved and the time 
        returned is the maximum simulation time. 
        The simulations always begin with the rover in its first form, with the orientation aligned with the x-axis (eastward).

        Args:
            rover: the rover object to be used
            map_number: index for which map the scenario will use
            scenario_number: index for which positions the scenario will use
        Returns:
            d: the normalised distance of the rover from the target (normalised with original distance)
            T: the normalised time taken for the scenario to complete (normalised with optimal time, i.e., 
                                                                       going straight with max. velocity)
        """
        # Initialising the scenario
        position = SCENARIO_POSITIONS[map_number][scenario_number][0:2]
        sample_position = SCENARIO_POSITIONS[map_number][scenario_number][2:4]

        xmin = MIN_BORDER_DISTANCE
        ymin = MIN_BORDER_DISTANCE
        xmax = self.env.heightmap_sizes[map_number][1] - MIN_BORDER_DISTANCE
        ymax = self.env.heightmap_sizes[map_number][0] - MIN_BORDER_DISTANCE
        
        rover.reset(position)
        distance_vector = sample_position - rover.position
        original_distance = distance_vector.norm()
        min_time_possible = original_distance/MAX_VELOCITY

        if detailed_results is not None:
            detailed_results.add(map_number, scenario_number, {'x': position[0],
                                                               'y': position[1],
                                                               'mode': rover.current_mode,
                                                               'direction': rover.angle/np.pi*180,
                                                               'mode_efficiency': rover.mode_efficiency})

        # Runs the scenario for X number of timesteps, where X is the max time / the time increment
        for timestep in range(0, SIM_TIME_STEPS):
            rover_view, mode_view = self.env.extract_local_view(rover.position, rover.angle, map_number)
            rover.update_rover_state(rover_view, mode_view, distance_vector, original_distance)
            distance_vector = sample_position - rover.position
            
            if detailed_results is not None:
                detailed_results.add(map_number, scenario_number, {'x': rover.position[0],
                                                                   'y': rover.position[1],
                                                                   'mode': rover.current_mode,
                                                                   'direction': rover.angle/np.pi*180,
                                                                   'mode_efficiency': rover.mode_efficiency})
            
            current_distance = distance_vector.norm()
           
            # Check if the rover went out of bounds, if so return the maximum time and end the scenario
            if not ((xmin <= rover.position[0] <= xmax) and (ymin <= rover.position[1] <= ymax)):
                return current_distance/original_distance, MAX_TIME/min_time_possible

            # # Checks if the sample has been found, if so return [0, relative time needed] and end the scenario
            if current_distance <= SAMPLE_RADIUS:
                return 0, timestep*DELTA_TIME/min_time_possible

        # If the for loop ends with no result, return end distance and maximum time
        return current_distance/original_distance, MAX_TIME/min_time_possible

    def plot(self, chromosome, plot_modes = False, plot_mode_efficiency = False):
        """
        This function acts the same as the fitness function, but also plots some of 
        the results for visualisation.

        Args:
            chromosome: decision vector/chromosome to be tested
            plot_modes: plot modes over time for each scenario. Optional.
            plot_mode_efficiency: plot mode efficiency over time for each scenario. Optional.
        """
        self.plot_modes(chromosome)

        _, detailed_results = self.pretty(chromosome, verbose = False)
        
        _, ax = plt.subplots(MAPS_PER_EVALUATION, SCENARIOS_PER_MAP, figsize=(15,15))
        for map_id in range(MAPS_PER_EVALUATION):
            for scenario_id in range(SCENARIOS_PER_MAP):
                ax_plot = ax_for_plotting(ax, map_id, scenario_id)
                self._plot_trajectory_of_single_scenario(map_id, scenario_id, ax_plot, detailed_results)
        ax_for_plotting(ax, 0, 0).set_title('Rover trajectories', fontsize=16)
        plt.tight_layout()
        
        if plot_modes == True:
            _, ax = plt.subplots(MAPS_PER_EVALUATION, SCENARIOS_PER_MAP, sharex = True, sharey= True, figsize=(15,15))
            for map_id in range(MAPS_PER_EVALUATION):
                for scenario_id in range(SCENARIOS_PER_MAP):
                    ax_plot = ax_for_plotting(ax, map_id, scenario_id)
                    self._plot_modes_of_single_scenario(map_id, scenario_id, ax_plot, detailed_results)
            ax_for_plotting(ax, 0, 0).set_title('Rover modes', fontsize=16)
        plt.tight_layout()
            
        if plot_mode_efficiency == True:
            _, ax = plt.subplots(MAPS_PER_EVALUATION, SCENARIOS_PER_MAP, sharex = True, sharey= True, figsize=(15,15))
            for map_id in range(MAPS_PER_EVALUATION):
                for scenario_id in range(SCENARIOS_PER_MAP):
                    ax_plot = ax_for_plotting(ax, map_id, scenario_id)
                    self._plot_mode_efficiency_of_single_scenario(map_id, scenario_id, ax_plot, detailed_results)
            ax_for_plotting(ax, 0, 0).set_title('Rover mode efficiency', fontsize=16)
        plt.tight_layout()
                
        plt.show()

    def plot_modes(self, chromosome):
        """
        Plot the masks of the rover modes.
        
        Args:
            chromosome: decision vector/chromosome to be visualised.
        """
        _, ax = plt.subplots(1,NUMBER_OF_MODES, figsize = (10,20))
        
        masks = chromosome[:NUMBER_OF_MODES*MASK_SIZE**2]
        masks = np.reshape(masks, (NUMBER_OF_MODES, MASK_SIZE, MASK_SIZE))
        
        for i in range(NUMBER_OF_MODES):
            ax[i].imshow(masks[i], cmap = 'terrain')
            ax[i].set_xticks([])
            ax[i].set_yticks([])
        ax[0].set_title('Rover mode masks', fontsize=10)
    
    def _plot_trajectory_of_single_scenario(self, map_id, scenario_id, ax, detailed_results):
        """
        Plots rover trajectory of a specific scenario using recorded data.
        
        Args:
            map_id: Map ID of the scenario.
            scenario_id: Scenario ID.
            ax: matplotlib ax for plotting.
            detailed_results: recorded results.
        """
        ax.imshow(self.env.heightmaps[map_id].flip(0), origin='lower', cmap = 'terrain')

        x_data = detailed_results[map_id][scenario_id]['x']
        y_data = np.array(detailed_results[map_id][scenario_id]['y'])
        ax.plot(x_data, y_data, color = 'k', linewidth= 1.5)

        # Plot the start, end and sample markers
        ax.plot(SCENARIO_POSITIONS[map_id][scenario_id][0],
                SCENARIO_POSITIONS[map_id][scenario_id][1], marker='o', markerfacecolor='blue')
        ax.plot(x_data[-1], y_data[-1], marker='o', markerfacecolor='orange')
        ax.plot(SCENARIO_POSITIONS[map_id][scenario_id][2],
                SCENARIO_POSITIONS[map_id][scenario_id][3], marker='d', markersize = 10, markerfacecolor='red')
        
        ax.set_xticks([])
        ax.set_yticks([])

    def _plot_modes_of_single_scenario(self, map_id, scenario_id, ax, detailed_results):
        """
        Plots rover modes during a specific scenario using recorded data.
        
        Args:
            map_id: Map ID of the scenario.
            scenario_id: Scenario ID.
            ax: matplotlib ax for plotting.
            detailed_results: recorded results.
        """
        y_data = detailed_results[map_id][scenario_id]['mode']
        ax.plot(y_data, color = 'k', linewidth= 1)
        ax.set_yticks(range(NUMBER_OF_MODES))
    
    def _plot_mode_efficiency_of_single_scenario(self, map_id, scenario_id, ax, detailed_results):
        """
        Plots rover mode efficiency of a specific scenario using recorded data.
        
        Args:
            map_id: Map ID of the scenario.
            scenario_id: Scenario ID.
            ax: matplotlib ax for plotting.
            detailed_results: recorded results.
        """
        y_data = detailed_results[map_id][scenario_id]['mode_efficiency']
        ax.plot(y_data, color = 'k', linewidth= 1)
        
udp = morphing_rover_UDP()