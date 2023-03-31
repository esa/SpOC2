## Wormhole traversal challenge
This challenge is part of the [SpOC (Space Optimisation Challenge)](https://www.esa.int/gsp/ACT/projects/spoc-2023) hosted by ESA's Advanced Concepts Team in partnership with [GECCO 2023](https://gecco-2023.sigevo.org/HomePage).

### Description
In the distant future, the dream of humankind to travel among the stars is about to become reality. The discovery of a habitable planet (called New Mars) with a breathable oxygen-nitrogen atmosphere orbiting Proxima Centauri is followed by an even more stunning discovery of a cluster of **black holes** close to the Solar System. The exciting part is that the black holes seem to be supporting a stable wormhole to distant corners of the Milky Way. Was it built by an unknown and unimaginably powerful alien civilisation, or is it a natural phenomenon? We may never know, but this does not stop us from preparing for the adventure of our life - the exploration of the newly discovered planet is finally within reach!

Twelve ships designed to withstand the colossal strain associated with traversing a wormhole are constructed and dispatched to the new planet via the wormhole network. The ships are dispatched together in order to avoid any surprises and delays, but it turns out that navigating a wormhole network is a lot more treacherous than anticipated: upon entering the region with the black holes, the twelve ships are quickly separated and pulled in different directions! Now each ship is facing the prospect of navigating the network on its own. *Danger, Will Robinson!*

To make matters worse, the effects of being close to an event horizon are already being felt: time is flowing at a different pace for each ship! What had started as a synchronised fleet is now at risk of being pulled apart further and further in time until they can no longer rendezvous at Proxima.

Luckily, every cloud has a silver lining: the physical properties of each black hole induce specific tension in the space-time continuum, which translates into a **probabilistic temporal offset** with a characteristic **mean** and **variance** for each wormhole associated with it. Importantly, the mean temporal offsets can also be **negative** - in other words, the wormholes can send the ships **back in time**! The twelve ships have the opportunity to harness these temporal offsets in order to arrive at their destination **within a given temporal window** with **minimal uncertainty (variance)**.

The arrival time for each ship is relative: it is the sum of the initial delay induced by the event horizon and the means of the temporal offsets of all wormholes that it traverses along the way (so negative means *subtract* time from the ship's arrival time). The variances are unfortunately all positive. This gives the ships lots of opportunities to make sure that they can fit within the specified temporal window, but they have to minimise the uncertainty.

While the quantum computers onboard the ships provide a pretty accurate estimate of the distribution of means and variances throughout the wormhole network, they are still unable to provide a feasible path through the network. With your expertise, the fleet still has a chance to make it all the way to Proxima unscathed!

Of course, there is one important detail that we saved for last: although the ships are designed to withstand the stress of traversing a wormhole, they still suffer minor damage each time they dip below the event horizon. To guarantee the structural integrity of the ships, you must make sure that each ship navigates to the destination within a certain number of hyperspace jumps.

Help find a feasible way through the wormhole network and save our intrepid adventurers from a fate of being forever lost in the mists of time!

### Data

We provide a [database](https://api.optimize.esa.int/data/spoc2/wormholes/database.npz) in the form of a compressed `numpy` archive named `database.npz`. It contains the necessary data to define a sparse **directed** network with `10000` nodes and `1998600` edges, as well as the initial conditions of the problem. In the context of the challenge, the nodes represent black holes and the edges represent wormholes. If an edge exists from node $a$ to node $b$, this does *not* imply that there is an edge from $b$ to $a$.

More specifically, the database contains the following entries:

- `edges`: `1998600` pairs of integers defining the edges of the network:

| Source node | Target node |
| ----------- | ----------- |
| 1           | 101         |
| 1           | 102         |
| ...         | ...         |
| 8374        | 9919        |

- `meanvar`: `1998600` pairs of floating-point numbers representing mean and variance of each edge:

| Edge         | Mean        | Variance   |
| ------------ | ----------- | ---------- |
| 1 -> 101     | -0.00120978 | 0.00278519 |
| 1 -> 102     | 0.00139366  | 0.00206549 |
| ...          | ...         | ...        |
| 8374 -> 9919 | 0.00014143  | 0.00102509 |

In the context of the challenge, the mean is a **relative temporal offset** associated with traversing the wormhole, and the variance is a measure of the uncertainty thereof. In other words, when a ship traverses a wormhole, the mean of that wormhole is added to its 'subjective time'. Specifically, a negative mean implies that the wormhole is sending the ship *backward in time*.

- `jump_limit`: The maximum number of jumps that any ship is allowed to make through the wormhole network.

- `window`: The temporal window within which all ships must arrive at the destination.

- `origins`: The IDs of the nodes that ships $0$ to $11$ (**in that order**) are allowed to use as entry points to the wormhole network.

- `destination`: The ID of the node where all $12$ ships must rendezvous.

- `delays`: A list containing the initial temporal offsets of ships $0$ to $11$, **in that order**.

The evaluation code (accessible under the `Evaluation code` section of the [problem page](https://optimize.esa.int/challenge/spoc-2-wormhole-transportation-network/p/wormhole-transportation-network) provides a convenience method (`_load_database()`) that loads the database and sets the attributes of the UDP. By default, this path is set to `./data/spoc2/wormholes/database.npz`.

### Encoding

The decision vector (*chromosome*) $x$ has a total length of $6000$ representing the paths traversed through the network by ships $0$ to $11$, **in this order**. Correct evaluation of the UDP depends on this assumed order because the first delay the `delays` list applies to the first ship, the second to the second ship, and so forth. The same applies to the `origins` array.

Section $i$ of the chromosome is of the form:

$$
x_i = [n_{i,0}, n_{i,1}, n_{i,2},..., n_{i,499}],
$$

where $n_{i,k} \in [0, 10000]$ is an `int` representing the black hole visited at jump $k$ by ship $i$. Therefore, the chromosome $x$ is a `1D` array representing the *concatenation* of the $12$ sections $x_{i}$:

$$
x = [x_{0}, ..., x_{11}]
$$

A value of $0$ has a special meaning. Because the chromosome has a fixed length, paths shorter than the maximum can be encoded by padding the path with $0$s to the required length of $500$. Therefore, if $x_{i}$ contains any $0$s, only the portion before the first $0$ is interpreted as the path for ship $i$.

NOTE: The UDP provides a convenience function called `convert_to_chromosome()`, which takes a potentially ragged list of lists of integers and returns a chromosome of the required length $6000$. The lists of integers are trimmed to a length of $500$ if they are longer than that or padded with $0$s if they are shorter.

## Constraints

The problem has **two equality constraints** and **one inequality constraint**.

### **Origin nodes**

The *first* node $n_{i,0}$ in each section $x_{i}$ must belong to the set of origin nodes for ship $i$:

$$
\forall i \in [0,11]: n_{i, 0} \in O_{i},
$$

where $O_{i}$ is the set of origin nodes for ship $i$. The origin nodes are defined in the database (cf. [Data](#data)).

This is the first equality constraint: `origin_node_constraint = udp.fitness(x)[1]`. A value of `0` indicates that this constraint is satisfied for all paths. A positive value indicates the number of paths for which the constraint is not satisfied.

### **Valid path**

Each pair of nodes $(n_{i,k}, n_{i,k+1}), k \in [0, 498]$ in $x_{i}$ must be an existing edge in the network. Additionally, $(n_{i,499}, t)$ must be a valid edge in the network:

$$
\forall i \in [0,11], \forall k \in [0, 498]: (n_{i, k}, n_{i, k+1}) \in E
$$

$$
\forall i \in [0,11]: (n_{i,499}, d) \in E,
$$

where where $d$ represents the destination node and $E$ is the set of existing edges in the network. Both the destination node and the existing edges are defined in the database (cf. [Data](#data)).

This is the second equality constraint: `valid_path_constraint = udp.fitness(x)[2]`. A value of `0` indicates that this constraint is satisfied for all paths. A positive value indicates the number of paths for which the constraint is not satisfied.

### **Arrival window**

All ships must arrive within a window of $1$ of each other (arbitrary units):

$$
\forall i,j \in [0,11]: \max_{i,j}(|T_{i} - T_{j}|) \leq 1,
$$

where $T_{i}$ is the **expected arrival time** of ship $i$ defined as the initial delay of that ship and the sum of the means $\mu_{i,k}$ of all edges (wormholes) along the path $x_{i}$ traversed by ship $i$, including the final jump to the destination node:

$$
\forall i \in [0,11]: T_{i} = D_{i} + \sum_{k}{\mu_{i,k}},
$$

where $D_{i}$ is the delay for ship $i$. The delays for all ships are defined in the database (cf. [Data](#data)).

This is the only inequality constraint: `arrival_window_constraint = udp.fitness(x)[3]`. The value represents the *maximum* of all the differences of the arrival times for all pairs of ships. A value `<=0` indicates that this constraint is satisfied for all paths.

### Objective

The objective of the challenge is to **minimise the maximum sum of variances** of the arrival times for **any** ship, subject to the above constraints.

$$
\min_{x}J = \min(\max_{i}{\sum_{k}{\nu_{i,k}}})
$$

where $J$ is the cost of the path traversal and $\nu{i, k}$ is the variance of edge $k$ along the path taken by ship $i$.

### Utilities / Hints

- Have a detailed look at the [User Defined Problem (UDP) ](https://optimise.esa.int/challenge/spoc-2-wormhole-transportation-network/p/wormhole-transportation-network) accessible under `Evaluation code` to understand how the evaluation works.

- This problem has a single objective (cf. [Objective](#objective)). Please submit the decision vector for the `decisionVector` key of the JSON submission file:

```json
[
    {
            "decisionVector": x,
            "problem": "wormhole-transportation-network",
            "challenge": "spoc-2-wormhole-transportation-network",
    }
]
```

- To decode the information from a decision vector `x` into a summary printed on screen, call the `.pretty(x)` method.

- Basic plots for solution visualisation purposes can be obtained by calling the `.plot(x)` method of the UDP (`x` is a chromosome).

- An example chromosome is provided in the source code and can be evaluated by using the `.example()` method of the UDP. Please note that this solution corresponds to a minimal working example and should not be taken as a baseline for what the winning score should be.

- For any decision vector `x`, the vector containing the fitness value and the equality and inequality constraint violations can be retrieved by using the `.fitness()` method of the UDP.

- Keep in mind that the means of the wormholes can be **negative**, so ships can compensate for their initial delays by going not only forward, but also *backward* in time.

### Submitting your solution to Optimise

To submit a solution (a chromosome), you can prepare a submission file by using the [submission helper](https://api.optimize.esa.int/data/tools/submission_helper.py) via

```python
from submisson_helper import create_submission
create_submission("spoc-2-wormhole-transportation-network","wormhole-transportation-network",x,"submission_file.json","submission_name","submission_description")
```

and [submit it](https://optimize.esa.int/submit).

### Dependencies

The evaluation code has been written and tested in Python 3.10.8 and depends on the following libraries:

* networkx >= 3.0.0
* numpy >= 1.24.2
* matplotlib >= 3.6.3
* loguru >= 0.6.0