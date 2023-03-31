This challenge is part of the [SpOC (Space Optimisation Challenge)](https://www.esa.int/gsp/ACT/projects/spoc-2023)
organised by ESA's Advanced Concepts Team and hosted in [GECCO 2023](https://gecco-2023.sigevo.org/HomePage).

## Morphing Rovers

### Description

Surface probes have detected vital information samples for assessing the habitability of New Mars. However,  there is only limited time before the samples expire. Since a crewed mission to the surface is still too dangerous, the fleet has decided to send highly advanced autonomous rovers to recover the samples as quickly as possible. Facilities on board the orbiting starships can manufacture and land these rovers rapidly, but given the unique terrain on this planet, previous rover designs are not applicable.

The advanced rover that will be sent is a new prototype known as a *morphing* rover, as it is capable of radically altering its shape between 4 different, pre-programmed forms. These forms are adapted for specfic terrains and as a result, their speed is dependant on the local terrain, e.g., one form could be ball-shaped for rolling downhill. The rovers are controlled by a neural network that determines both the direction in which to travel as well as when to morph, based on acquired data about the local terrain and mission conditions.

30 samples are located across 6 different regions, with 5 samples in each region. The only possible landing site for each individual sample has also been determined. Given the urgency of the mission, a single optimal rover configuration must be developed. One copy of this design will be sent to each landing site and make its way to a designated sample. By the time the rovers land, the samples will only have 500 minutes remaining before expiring. If the rovers do not reach the samples in time or leave the region where communication with the fleet is possible, they fail in their mission. This optimal rover design must be developed immediately and the brightest minds in the fleet have been assembled to tackle this challenge.

Your task is to design both the morphing forms and neural network configuration that allow the rover to optimally navigate the terrain of New Mars and recover the samples. Your objective is to maximise the number of samples retrieved and to minimise the time taken to do so.

### Data

You are provided with topological [maps](https://api.optimize.esa.int/data/spoc2/morphing/Maps/) of the 6 regions, best interpreted as scalar fields, with the values being the altitude of the terrain at a point. In addition, you are also provided with the [coordinates](https://api.optimize.esa.int/data/spoc2/morphing/coordinates.txt) of the landing sites as well as sample locations for each recovery scenario.

The topological map for each region is identified by a name, e.g., `Map1`, and stored as an 8-bit JPG file. The landing site and sample coordinates are stored as a list of lists of the form $[X, L_x, L_y, A_x, A_y]$, where X is the number of the map the recovery takes place in, $L_x$, $L_y$ are the coordinates of the landing site, and $A_x$, $A_y$ are the coordinates of the samples.

**In order to use the evaluation code provided with the problem**, the coordinates database must be named `coordinates.txt` and located in the path specified by the `PATH` variable in the `Evaluation code` (Line 29).  By default, this path is set to `.\data\spoc2\morphing`.
Additionally, the topological maps must be named `MapX.JPG` (where X is a number from 1-6) and located in a folder called `Maps` in `PATH`. By default, this yields `.\data\spoc2\morphing\Maps\MapX.JPG` for each map.

### Simulation

A rover is defined by two main components: the 4 different forms it can morph into as well as a neural network deciding when to change form and in which direction to drive.

Each form is modelled via an $11\times 11$ matrix (or mask) $F^{(i)}$ ($i \in [0,3]$), representing the terrain morphology the rover form is adapted to (hence, we are not directly designing the rover shape of each form, but the type of terrain the form moves quickly on). To obtain the current magnitude of the rover velocity, the active mask is compared to the terrain $T$ the rover is standing on (again a $11\times 11$ matrix) using a velocity function $\nu = V(F^{(i)}, T)$, $\nu \in [0,1]$. 

During the simulation, the position $r$ of the rover is then updated: 
$$r(t+1) = r(t) + \nu \cdot v_\text{max} \cdot e_\alpha \,,$$

where $e_\alpha = (\text{cos}(\alpha), \text{sin}(\alpha))$ is the unit vector pointing in the direction $\alpha$ the rover is looking and $v_\text{max}$ is the maximum velocity achievable by the rover.

The rover is controlled by a neural network $\Phi$, taking in an $89\times 89$ cut-out of the terrain $M$ around the rover (taken from orbit, i.e., always same orientation), the previous activity of the last hidden layer in the network $h$ and the state vector $s$ of the rover with elements being:

* the current velocity factor $\nu$
* the ratio of current morphing cooldown and maximum cooldown
* the angular difference from rover orientation to target location 
* ratio of current and start distance to target location
* current orientation
* current mode (one-hot coded)

The neural network $\Phi(M, h, s)$ reads in the surrounding terrain $M$ using two convolutional layers and the state $s$ using dense layers. $h$ is fed into the last hidden layer using recurrent weights. As output, the network provides a real-valued scalar $m_s$ that controls mode switching as well as an angular velocity $\omega$ which is limited between -PI/4 and PI/4. The rover orientation $\alpha$ is then updated as 
$$\alpha(t+1) = \alpha(t) + \omega \,.$$

If $m_s >0$ and the cooldown is $0$, then the rover switches to the mode that is best for the current terrain and the cooldown is activated.

For each scenario, the rover is simulated for 500 time steps, i.e., this is the maximum time window in which the rover can reach the target. If the rover reaches the target, the simulation is stopped prematurely. If the rover leaves the map, the simulation directly jumps to the last time step. The time required for the rover to reach the target is given by the number of time steps until the simulation is terminated.

### Encoding

The decision vector $x$, also called *chromosome*, consists of 3 parts:

* Parameters of the 4 rover-form masks $[F^{(0)}_{0,0}, F^{(0)}_{0,1},  . . . , F^{(3)}_{10, 9}, F^{(3)}_{10, 10}]$, where $F^{(i)}_{j,k}$ is the value of mask $i$ at indices $j,k$. All values are `float` variables in $[-100, 100]$ except the central element of each mask, which are `float` variables in $[10^{-16}, 100]$.
This part must contain `484` parameters ($11\times 11$ masks over 4 rover-forms). The rover is always initialized in the first mask. The order of the masks here corresponds to the index the rover uses to select masks.

* Parameters of the neural network $[b_1, ..., b_N, W_1, ..., W_M]$, where the first part are the biases $b_i$ and the second part the weights $W_i$. In total, there are `N=146` bias parameters and `M=18496` weight parameters, totalling `18642` parameters. The parameters are all float values in $[-100, 100]$.
    * The biases further split up into:
        * First 40: biases of hidden neurons `h1` that get the rover state as input $s$.
        * Next 8: first convolutional layer `conv1`.
        * Next 16: second convoltional layer `conv2`.
        * Next 40: hidden neurons `h2` getting input from `h1` and `conv2`.
        * Next 40: hidden neurons `h3` getting input from `h2` and themselves (recurrent connections).
        * Next 2: output neurons `output` getting input from `h3`.
    * The weights split up into:
        * First 360: weights from $s$ to `h1` (original shape: $[40, 9]$, i.e., from input dimension $10$ to hidden dimension $40$).
        * Next 968: from $M$ to `conv1` ($[8, 1, 11, 11]$, i.e., 8 convolutional filter, 1 input channel, $11\times 11$ filter size).
        * Next 2048: from `conv1` to `conv2` ($[16,8,4,4]$).
        * Next 10240: from `conv2` to `h2` ($[40, 256]$).
        * Next 1600: from `h1` to `h2` ($[40,40]$).
        * Next 1600: from `h2` to `h3` ($[40,40]$).
        * Next 1600: from `h3` to `h3` (recurrent connection, $[40,40]$)
        * Next 80: from `h3` to `output` ($[2,40]$).
* Additional hyperparameters for the neural network: $[p_1, p_2, a_1, a_2, a_3, a_4, a_5]$, all values being integers.
    * $p_1$: type of pooling layer after the 1st convolution layer ($0$ or $1$).
        * $p_1=0$: MaxPool2D
        * $p_1=1$: AvgPool2D
    * $p_2$: type of pooling layer after the 2nd convolution layer ($0$ or $1$).
        * same options as $p_1$
    * $a_1$: activation function of the 1st convolutional layers.
        * $a_1 = 0$: Sigmoid
        * $a_1 = 1$: Hard Sigmoid
        * $a_1 = 2$: Tanh
        * $a_1 = 3$: Hard Tanh
        * $a_1 = 4$: Softsign
        * $a_1 = 5$: Softplus
        * $a_1 = 6$: ReLu
    * $a_2$: activation function of the 2nd convolutional layers.
        * same options as $a_1$
    * $a_3$: activation function of `h1` layer.
        * same options as $a_1$
    * $a_4$: activation function of `h2` layer.
        * same options as $a_1$
    * $a_5$: activation function of `h3` layer.
        * same options as $a_1$, but only takes values between 0 and 4.

### Objective

The objective of this challenge is to maximise the number of samples recovered and minimise the total time required to do so $T(x)$ with chromosome $x$. This is measured using the following fitness function $f$:

$$
f(x) = \frac{T}{T_\text{min}} \cdot \left(1+\frac{d(x)}{d_0}\right)
$$

where $T_\text{min}$ is the fastest possible time for the rover to get to its target (i.e., driving in a straight line with maximum velocity). $d_0$ is the original distance to the target at simulation start, and $d(x)$ is the rover's distance to the target at the end of the simulation. If the target is touching the rover ($d(x) \leq 6$), $d(x)$ is set to $0$ instead. The task is to find an $x$ that minimizes $f$ over all scenarios:

$$
\min_{x} \frac{1}{\text{no.\ scenarios}} \sum_\text{scenarios} f(x)
$$

For any decision vector $x$, the fitness can be retrieved with the following line of code:

```
f = udp.fitness(x)
```

### Submitting

To submit a solution, you can prepare a submission file with the [submission helper](https://api.optimize.esa.int/data/tools/submission_helper.py) via

```python
from submisson_helper import create_submission
create_submission("spoc-2-morphing-rovers","morphing-rovers",x,"submission_file.json","submission_name","submission_description")
```

and [submit it](https://optimize.esa.int/submit). After submitting a solution, it will be evaluated immediately. For this challenge, **this process can take a bit of time, so please be patient** and wait until the fitness of your submitted solution is returned.

### Utilities / Hints

* We will be using [GitHub](https://github.com/esa/SpOC2) as our hub for communication with competitors. Our primary means of communication will be the Discussions feature on our repository.

* Have a detailed look at the [User Defined Problem (UDP) ](https://optimize.esa.int/challenge/spoc-2-morphing-rovers/p/morphing-rovers) accessible under `Evaluation code` to understand how the evaluation works.

* You can set the file path to the data on your system using the global variable `PATH` in the `Evaluation code` .

* Plots for the rover forms and the rover trajectory taken on each scenario/map can be obtained by calling `udp.plot(x)`. Calling `udp.plot(x, plot_modes=True, plot_mode_efficiency=True)` will return the same results as well as plots detailing the form changes and form efficiency for each scenario/map.

* Detailed results per scenario/map can be obtained by calling `udp.pretty(x)`. This will print a table of all scores, as well as return the mean score and a data structure containing detailed results about rover trajectories, velocities, etc.

* An example chromosome is provided in the source code which can be loaded by calling `udp.example()`. If you want to evaluate the example chromosome, call `udp.fitness(x)` with `x = udp.example()`. Please note that this solution corresponds to a working example for some of the scenarios and should not be taken as a baseline for what an optimal configuration should be.

* The number of neural network parameters, mask size, etc. are available as constants `NUM_BIASES`, `NUM_WEIGHTS`, `NUM_MODES`, `MASK_SIZE` in the solution script.

* The position of the central mask elements in the chromosome are stored in the global variable `MASK_CENTRES`.

* By changing `MAPS_PER_EVALUATION` and `SCENARIOS_PER_MAP`, you can choose a smaller set of scenarios to train or test on.

* You can add your own scenarios by adding sets of coordinates to `coordinates.txt`, and maps to the `/Maps` folder. You have to add these to  `HEIGHTMAP_NAMES` as well as change `MAPS_PER_EVALUATION` and `SCENARIOS_PER_MAP` accordingly.

* The rover class only requires a valid chromosome $x$ to construct a valid rover.

* Both the rover class and the neural network class have an attribute `chromosome` returning their chromosomes.

* You can use `udp.env.extract_local_view(position, direction_angle, map_id)` to extract the surrounding terrain and the terrain the rover is standing on given a rover position, its orientation (in [rad]) and the ID of the used map.

* The function `velocity_function(form, mode_view)` calculates the efficiency of the rover given the mask of a form and a terrain.

* Gradient calculation in the neural network is by default turned off. Remove `self._turn_off_gradients()` in the init if you want to train the network using gradient descent.

* You can initially set the weights and biases manually for the output neurons in the neural network, e.g., for enabling permanent switching between modes, or to turn off switching altogether.

### Acknowledgments

The heightmaps of Mars were taken from the [MOLA Mission Experiment Gridded Data Records (MEGDRs)](https://pds-geosciences.wustl.edu/missions/mgs/megdr.html), made publicly available by NASA.