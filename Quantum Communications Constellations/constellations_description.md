This challenge is part of the [SpOC (Space Optimisation Challenge)](https://www.esa.int/gsp/ACT/projects/spoc-2023)
organised by ESA's Advanced Concepts Team and hosted in [GECCO 2023](https://gecco-2023.sigevo.org/HomePage).

## Quantum Communications Constellations

### Description
Finally, after a long and perilous journey, the motherships reach New Mars. As the settlement effort on the planet begins, one of the key challenges is establishing reliable communication channels between the planet's surface, where rovers will be deployed, and the motherships in orbit. To tackle this challenge, teams of engineers have been working tirelessly to design and deploy quantum communications networks in orbit.

By leveraging laser-based (line-of-sight) communication, constellations of quantum satellites provide provably-secure infrastructure to support the rovers. Nonetheless, the efficiency of a quantum link evolves with the distance between the communicating satellites. In addition, the deployment of such state-of-the-art communications hardware comes with significant operational costs. In determining the number of satellites to deploy and their orbital distributions, the engineers face a non-trivial dilemma: either maximise communication efficiency at the cost of demanding operational resources or minimise the number of satellites and risk having unreliable network infrastructure. Having recognised the underlying tradeoff, the engineers resort to constellation design optimisation to resolve it.

After much deliberation, the engineers agreed that two Walker Delta Pattern constellations would provide ample support for the next 10 years of planned rover operations. However, they still need help finding optimal, collision-free orbital configurations to allow each constellation to reliably relay transmissions from the motherships to the surface. Specifically, the number of planes per constellation, the number of quantum satellites per plane, their semi-major axis, inclination, eccentricity and phasing all still need to be determined. 

Fortunately, the engineers have been granted some leeway to choose the starting locations of the rovers which will carry out the surface operations. Out of a possible 100 rover placements, they are tasked with identifying 4 non-collocated rover starting locations that maintain good reception of mothership transmissions over a 10 year period.

While the deployment of reliable communication in orbit is a crucial step of the New Mars mission, the engineers are convinced that only the community will know how to best make use of the limited resources to bring it to fruition.

### Data

We provide a small database of 100 possible rover positions in the file [rovers.txt](https://api.optimize.esa.int/data/spoc2/constellations/rovers.txt). 
The file has the following format:

|Lat (rad.)      | Lon (rad.) | 
|--------------------|---------------------|
|3.290891754147420301e-01| 3.455813102533720649e+00|
|9.210033238890864560e-01| 1.222549040618014837e+00|
|...........|.............|

The selection of 4 candidate positions for rover deployment is carried out by encoding in the decision vector the corresponding line $i\in[0, 99]$ , as detailed below.

NOTE: the [python code provided](https://api.optimize.esa.int/media/problems/spoc-2-quantum-communications-constellations-quantum-communications-constellatio_DhZfx09.py) for the User Defined Problem (udp) expects to find a file ```rover.txt``` located in the local directory ```./data/spoc2/constellations/rovers.txt"```.
This path can be modified in the udp python file to match a specific local installation.

### Encoding

The decision vector `x`, also called *chromosome*, is split into:

* Orbital parameters of the first Walker constellation: <br>
`[a1, e1, i1, w1, eta1]`<br>
Where
    * `a1` $\in [1.06, 1.8]$ is a `float` representing the semi-major axis (in km).
    * `e1` $\in [0, 0.02]$ is a `float` representing the eccentricity.
    * `i1` $\in [0, \pi]$ is a `float` representing the inclination (in rad).
    * `w1` $\in [0, 2\pi]$ is a `float` representing the argument of perigee (in rad).
    * `eta1` $\in [1, 1000]$ is a `float` defined as the quality indicator $\eta$ common to all satellites in the first Walker constellation.

* Orbital parameters of the second Walker constellation: <br>
`[a2, e2, i2, w2, eta2]`<br>
Where
    * `a2` $\in [2, 3.5]$ is a `float` representing the normalised semi-major axis (in km).
    * `e2` $\in [0, 0.1]$ is a `float` representing the eccentricity.
    * `i2` $\in [0, \pi]$ is a `float` representing the inclination (in rad).
    * `w2` $\in [0, 2\pi]$ is a `float` representing the argument of the perigee (in rad).
    * `eta2` $\in [1, 1000]$ is a `float` defined as the quality indicator $\eta$ common to all satellites in the second Walker constellation.

* Parametrization of the first Walker constellation: <br>
`[S1, P1, F1]`<br>
Where 
    * `S1` $\in [4, 10]$ is an `integer` corresponding to the number of satellites per plane.
    * `P1` $\in [2, 10]$ is an `integer` corresponding to the number of planes.
    * `F1` $\in [0, 9]$ is an `integer` defining the phasing of the constellation. For instance, `F1 = N` means the phasing of the constellation repeats every $N$ planes.

* Parametrization of the second Walker constellation: <br>
`[S2, P2, F2]`<br>
Where 
    * `S2` $\in [4, 10]$ is an `integer` corresponding to the number of satellites per plane.
    * `P2` $\in [2, 10]$ is an `integer` corresponding to the number of planes.
    * `F2`  $\in [0, 9]$ is an `integer` defining the phasing of the constellation. For instance, `F2 = N` means the phasing of the constellation repeats every $N$ planes.

* Rover surface positioning: <br>
`[r1, r2, r3, r4]`<br>
Where
    * `r1` $\in [0, 99]$ is an `integer` corresponding to the first rover (line number in the database of rover positions).
    * `r2` $\in [0, 99]$ is an `integer` corresponding to the index of the second rover (line number in the database of rover positions).
    * `r3` $\in [0, 99]$ is an `integer` corresponding to the index of the third rover (line number in the database of rover positions).
    * `r4` $\in [0, 99]$ is an `integer` corresponding to the index of the fourth rover (line number in the database of rover positions).

The complete decision vector `x` is therefore defined as:

`x = [a1, e1, i1, w1,  eta1] + [a2, e2, i2, w2,  eta2] + [S1, P1, F1] +[S2, P2, F2] + [r1, r2, r3, r4]`


### Constraints
1. Surface cover: the minimum distance between any two selected rovers position  needs to be at least `3000km` to ensure a good coverage of the planetary surface. 
Implemented as an inequality constraint: compute as `rover_constraint= udp.fitness(x)[3]`. Satisfied when negative.

2. Collision avoidance: two satellites can only be considered as part of the communication path if they are at least `50km` apart; this applies to both inter and intra-constellation satellite communications.  
Implemented as an inequality constraint: compute as `collision_constraint= udp.fitness(x)[4]`. Satisfied when negative.

### Objectives

This optimisation problem has two objectives:

1. **Minimising the average shortest communication cost** between the 7 motherships and the 4 rovers (via the Walker constellation satellites).

    To be feasible, a communications path must start with a mothership (source node) and end with a rover (destination node), and it may pass through $N-2$ nearby quantum satellites (intermediary nodes) which all have line-of-sight. Additionally, the zenith angle $\theta_Z$ of the communications link reaching the rover must satisfy $\theta_Z \leq \pi / 3$. Then, the communications cost can be expressed as a sum over $N-1$ links:
    $$
    QC = \sum_{i=1}^{N-1}\big(-\log(\eta_{i+1})  + 2\log(L_{i})\big) + \frac{1}{\sin(\pi/2 - \theta_Z)}
    $$
    where
    * $\eta_{i}$ is the quality indicator of the $i^{th}$ satellite in the chain. It will be numerically equal to `eta1` or `eta2` according to the Walker constellation the $i^{th}$  satellite belongs to.
    * $L_i$ is the distance between the $i^{th}$ and ${i+1}^{th}$ satellites in the chain, in km.
    * $\theta_Z$ is the zenith angle of the communications link for the final satellite-rover link.

    Denoting with QC$^*$ its minimum value (across all feasible paths), the first objective is the average 
    of QC$^*$ across all mother-ship rovers pairs and ten years.
    $$
    J_1 = \overline{QC^*}
    $$

2. **Minimising the total cost of  manufacturing, launching and operating both constellations.**
Such cost is defined as a function of the number of satellites put in orbit as well as their quality metric:
    $$
    J_2 = \eta_1 S_1 P_1 + \eta_2 S_2 P_2
    $$

These two objectives are in conflict with each other, as a good average shortest communication path ($J_1$) requires many satellites and a large $\eta$. The opposite is true for $J_2$.

We are interested in the best possible trade-offs between those two objectives which also satisfies the aforementioned constraints. Consequently, you are tasked to submit a set of optimal trade-off solutions (e.g. a list of multiple different decision vectors) which should approximate the Pareto frontier of the problem. The quality of your approximated Pareto frontier will then be assessed by the [hypervolume indicator](https://esa.github.io/pygmo2/tutorials/hypervolume.html) with a reference point of $[1.2, 1.4]$ in objective space, which we then multiply by 10000. Thus, only solutions with costs $J_1$ less than `1.4` and $J_2$ less than `1.2` will contribute to the hypervolume. Note that both $J_1$ and $J_2$ are normalized by some factor as to have similar magnitude. 

###Submitting

To submit a solution, you can prepare a submission file with the [submission helper](https://api.optimize.esa.int/data/tools/submission_helper.py) via

```python
from submisson_helper import create_submission
create_submission("spoc-2-quantum-communications-constellations","quantum-communications-constellations",x,"submission_file.json","submission_name","submission_description")
```

and [submit it](https://optimize.esa.int/submit). 

### Utilities / Hints

* We will be using [GitHub](https://github.com/esa/SpOC2) as our hub for communication with competitors. Our primary means of communication will be the [Discussions](https://github.com/esa/SpOC2/discussions) feature on our repository.

* Have a detailed look at the [User Defined Problem (UDP) ](https://optimize.esa.int/challenge/spoc-quantum-constellations/p/quantum-constellations) accessible under `Evaluation code` to understand how the evaluation works.

* This problem is multi-objective and thus benefits from submitting multiple decision vectors corresponding to good trade-offs. You are allowed to submit up to 100 decision vectors for this problem. Assuming, you have 100 decision vectors $x_1, \ldots, x_{100}$, make sure to submit them as a list for the `decisionVector`-key of the JSON submission file:

    ```
    [
        {
                "decisionVector": [ x1, x2, ... , x100 ],
                "problem": "quantum-communications-constellations",
                "challenge": "spoc-2-quantum-communications-constellations",
        }
    ] 
    ```

* To decode the information from a decision vector `x` into a summary printed on screen, call `udp.pretty(x)`. 

* Basic plots for solution visualisation purposes can be obtained by calling `udp.plot(x)`.

* An example chromosome is provided in the source code and can be evaluated by calling `udp.example()`. Please note that this solution corresponds to a minimal working example and should not be taken as a baseline for what the winning score should be.

* For any decision vector `x`, both objectives and the constraint violations can be retrieved by the following line of code:

    ```python
    J1, J2, ineq_constraint1, ineq_constraint2 = udp.fitness(x)
    ```
