from typing import Set
from typing import List
from typing import Tuple
from typing import Union
from typing import Optional

# --------------------------------------
import sys

# --------------------------------------
from pathlib import Path

# --------------------------------------
import numpy as np

# --------------------------------------
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# --------------------------------------
import networkx as nx

# --------------------------------------
from loguru import logger

logger.remove()
logger.add(sys.stdout, format="{time:HH:mm:ss.SS} | {level} | {message}")

# --------------------------------------
# Type definitions
FigAxes = Tuple[Figure, np.ndarray]
PathList = List[List[int]]
FSPath = Union[Path, str]
EvalRetType = Tuple[List[float], Optional[FigAxes]]


class wormhole_traversal_udp:
    """
    UDP (User-Defined Problem) class for the Wormhole Traversal Challenge of the
    SpOC competition as part of the GECCO 2023 conference.

    Conceptually, we define a sparse network of black holes (nodes) and wormholes (edges).
    Each edge in the network is associated with a mean and a variance, both of which are cumulative.
    The challenge is to minimise the maximum variance of any of the 12 paths traversed across
    the network while satisfying the equality and inequality constraints.

    The formal specification of the problem and all constraints and objectives
    can be found on its page on the Optimise platform:

    https://optimise.esa.int/challenge/spoc-2-wormhole-transportation-network/About

    For more information about the competition, please refer to the competition website:

    https://www.esa.int/gsp/ACT/projects/gecco-2023-competition/

    This class conforms to the pygmo UDP format.

    ====[ ATTENTION ]====

        All IDs are 1-based!

    ====[ ATTENTION ]====
    """

    """
    ####################
      Private methods
    ####################
    """

    def __init__(
        self,
        database: FSPath = "./data/spoc2/wormholes/database.npz",
        n_ships: int = 12,
        jump_limit: int = 500,
        window: float = 1.0,
    ):
        # Number of ships
        self.n_ships: int = n_ships

        # Maximum number of jumps
        self.jump_limit: int = jump_limit

        # Arrival window
        self.window: float = window

        # A networkx DiGraph object.
        self.network: nx.DiGraph = None

        # Number of nodes (black holes)
        self.n_bh: int = 0

        # Number of edges (wormholes)
        self.n_wh: int = 0

        # The sets of origin nodes
        self.origins: List[Set[int]] = None

        # The destination node
        self.destination: int = 0

        # Ship delays
        self.delays: np.ndarray = None

        # * Convenience attributes * #
        # Used during the fitness evaluation.
        # The final return value is a concatenated version
        # of the fitness value, the equality constraints
        # and the inequality constraints.
        self._fitness = None
        self._eq_constraints = None
        self._iq_constraints = None
        self._all_constraints_satisfied = True
        self._scaling = 10000

        # Reset the fitness attributes
        self._reset_fitness_attributes()

        # Load the database and update
        self._load_database(database)

    """
    ######################
       Protected methods
    ######################
    """

    def _load_database(
        self,
        filename: FSPath,
    ) -> None:
        """
        Load the network from a compressed NumPy archive
        into a networkx DiGraph object.

        Args:
            filename (FSPath):
                Path to the database file.
        """

        try:
            filename = Path(filename)

            if not filename.exists():
                raise ValueError(
                    f"The specified file ({filename.absolute()}) does not exist."
                )

            loaded = np.load(filename)

            loaded_edges = loaded["edges"].astype(np.int32)
            loaded_meanvar = loaded["meanvar"].astype(np.float32)
            loaded_jump_limit = loaded.get("jump_limit", None)
            loaded_window = loaded.get("window", None)
            loaded_origins = loaded["origins"]
            loaded_destination = loaded["destination"]
            loaded_delays = loaded["delays"]

            # * Graph * #
            graph = nx.DiGraph()
            graph.add_edges_from(loaded_edges)

            self.network = graph

            # * Extract the number of nodes and edges * #
            self.n_bh = self.network.number_of_nodes()
            self.n_wh = self.network.number_of_edges()

            # * Mean and variance * #
            for idx, edge in enumerate(loaded_edges):
                graph[edge[0]][edge[1]]["m"] = loaded_meanvar[idx][0]
                graph[edge[0]][edge[1]]["v"] = loaded_meanvar[idx][1]

            # * Jump limit * #
            if loaded_jump_limit is not None:
                self.jump_limit = int(loaded_jump_limit.item())

            # * Window * #
            if loaded_window is not None:
                self.window = float(loaded_window.item())

            # * Origin and destination nodes * #
            ship_ids = list(range(1, self.n_ships + 1))
            origins = []
            for s_id, origin_list in zip(ship_ids, loaded_origins):
                origins.append(set(origin_list))
                for origin in origin_list:
                    graph.nodes[origin]["o"] = s_id

            self.origins = origins
            self.destination = loaded_destination.item()
            graph.nodes[self.destination]["d"] = True

            # * Delays * #
            # The delays are already in a NumPy array
            self.delays = loaded_delays.astype(np.float32)

            logger.info(
                f"Loaded a network with {self.n_bh} nodes and {self.n_wh} edges"
            )

        except Exception as e:
            raise ValueError(f"Error loading the database from {filename}:\n{e}")

    def _update_ecs(
        self,
        constraint: int,
    ) -> None:
        """
        Update the equality constraints.

        Args:
            constraint (int):
                An equality constraint (should be =0 to be considered satisfied).
        """
        self._eq_constraints.append(constraint)
        self._all_constraints_satisfied = self._all_constraints_satisfied and (
            constraint == 0
        )

    def _update_ics(
        self,
        constraint: float,
    ) -> None:
        """
        Update the inequality constraints.

        Args:
            constraint (float):
                An inequality constraint (should be <=0 to be considered satisfied).
        """

        self._iq_constraints.append(constraint)
        self._all_constraints_satisfied = self._all_constraints_satisfied and (
            constraint <= 0.0
        )

    def _reset_fitness_attributes(self) -> None:
        self._fitness = []
        self._eq_constraints = []
        self._iq_constraints = []
        self._all_constraints_satisfied = True

    def _compose_udp_retval(self) -> List[Union[int, float]]:
        """
        Helper method that performs some sanity checks before constructing
        the return value of the fitness() method, which consists of the fitness value(s),
        the equality constraints and the inequality constraints, in that order.

        Raises:
            ValueError:
                Error raised if the number of fitness values is wrong.

            ValueError:
                Error raised if the number of equality constraints is wrong.

            ValueError:
                Error raised if the number of inequality constraints is wrong.

        Returns:
            List[Union[int, float]]:
                The list of fitness value(s), equality constraints and inequality constraints.
        """

        if len(self._fitness) != self.get_nobj():
            raise ValueError(
                f"Wrong number of fitness values: expected {self.get_nobj()}, got {len(self._fitness)}."
            )

        if len(self._eq_constraints) != self.get_nec():
            raise ValueError(
                f"Wrong number of equality constraints: expected {self.get_nec()}, got {len(self._eq_constraints)}."
            )

        if len(self._iq_constraints) != self.get_nic():
            raise ValueError(
                f"Wrong number of inequality constraints: expected {self.get_nic()}, got {len(self._iq_constraints)}."
            )

        self._fitness.extend(self._eq_constraints)
        self._fitness.extend(self._iq_constraints)

        return self._fitness

    def _plot(
        self,
        x: Union[np.ndarray, PathList],
    ) -> FigAxes:
        """
        Plot the paths traversed by the ships to the destination.

        Args:
            x (Union[np.ndarray, PathList]):
                A chromosome or a list of paths taken by the ships.

        Returns:
            FigAxes:
                Matplotlib figure and axes.
        """

        if isinstance(x, np.ndarray):
            # Extract the paths from a chromosome
            paths = self._chromosome_to_paths(x)

        else:
            # Make a copy of the paths
            paths = [[n for n in path] for path in x]

        fig, ax = plt.subplots(2, 2, figsize=(10, 8))

        path_length_ax, path_mean_ax = ax[0][0], ax[0][1]
        cost_ax, window_ax = ax[1][0], ax[1][1]

        xs = list(range(1, self.n_ships + 1))

        # * Path lengths * #
        path_length_ys = [len(path) for path in paths]

        path_length_ax.bar(xs, path_length_ys, color="steelblue")

        path_length_ax.set_title("Total path length per ship")
        path_length_ax.set_xlabel("Ship")
        path_length_ax.set_ylabel("Path length")
        path_length_ax.set_xticks(xs)

        # * Path mean * #
        path_mean_ys = np.zeros((self.n_ships,))

        for ship, path in enumerate(paths):
            for src, tgt in zip(path[:-1], path[1:]):
                if src not in self.network.nodes or tgt not in self.network[src]:
                    break

                path_mean_ys[ship] += self.network[src][tgt]["m"]

        path_mean_ax.bar(xs, path_mean_ys, color="orangered")

        path_mean_ax.set_title("Total path mean per ship")
        path_mean_ax.set_xlabel("Ship")
        path_mean_ax.set_ylabel("Path mean")
        path_mean_ax.set_xticks(xs)

        # * Path cost (accumulated variance) * #
        cost_ys = np.zeros((self.n_ships,))

        for ship, path in enumerate(paths):
            for src, tgt in zip(path[:-1], path[1:]):
                if src not in self.network.nodes or tgt not in self.network[src]:
                    break

                cost_ys[ship] += self.network[src][tgt]["v"]

        cost_ys *= self._scaling

        cost_ax.bar(xs, cost_ys, color="magenta")

        cost_ax.set_title("Total path cost per ship")
        cost_ax.set_xlabel("Ship")
        cost_ax.set_ylabel("Cost")
        cost_ax.set_xticks(xs)

        # * Arrival window * #
        _, window_ys = self._compute_arrival_gaps(paths)

        window_ax.bar(xs, window_ys, color="forestgreen")

        margin = (self.window - (window_ys.max() - window_ys.min())) / 2

        window_ax.axhline(
            y=window_ys.max() + margin,
            color="#aaaaaa",
            linestyle="--",
            linewidth=0.5,
        )
        window_ax.axhline(
            y=window_ys.min() - margin,
            color="#aaaaaa",
            linestyle="--",
            linewidth=0.5,
        )

        window_ax.annotate(
            "",
            xy=(0, window_ys.min()),
            xycoords="data",
            xytext=(0, window_ys.max()),
            textcoords="data",
            arrowprops=dict(
                arrowstyle="<->",
                connectionstyle="arc3",
                color="r",
                lw=0.5,
            ),
        )
        window_ax.set_xticks(xs)
        window_ax.set_title("Relative arrival times")
        window_ax.set_xlabel("Ship")
        window_ax.set_ylabel("Relative arrival time")

        plt.text(
            -0.5,
            window_ys.min() + 0.2,
            f"Range: {window_ys.max() - window_ys.min()}",
            rotation=90,
            fontsize=8,
        )

        plt.xlim(-0.90, self.n_ships + 1)
        fig.tight_layout()

        return (fig, ax)

    def _chromosome_to_paths(
        self,
        x: np.ndarray,
    ) -> PathList:
        """
        Converts a chromosome into a list of (truncated) paths.

        Used in the fitness evaluation function.

        Args:
            x (np.ndarray):
                Chromosome.

        Returns:
            PathList:
                A list of paths.
        """
        # Extract the paths
        paths = []

        for idx in range(self.n_ships):
            # Extract the ship's path
            path = x[idx * self.jump_limit : (idx + 1) * self.jump_limit]

            # Check if the path contains 0s.
            # If it does, trim the path to that point.
            fz = np.argwhere(path == 0)
            if len(fz) > 0:
                path = path[: fz[0][0]]

            # Append the target node
            path = np.append(path, self.destination)

            # Add the path to the list of paths
            paths.append(path)

        return paths

    def _compute_arrival_gaps(
        self,
        paths: PathList,
    ) -> np.ndarray:
        """
        Check if all ships arrive at the destination within the window.

        Args:
            paths (PathList):
                Ship paths.

        Returns:
            np.ndarray:
                The pairwise distances.
        """
        arrival_times = np.array(
            [
                nx.path_weight(self.network, path, weight="m")
                if nx.is_path(self.network, path)
                else idx * self.window
                for idx, path in enumerate(paths)
            ],
            dtype=np.float32,
        )

        # Add the delays
        arrival_times += self.delays
        arrival_times = arrival_times[None, :]

        # Compute the pairwise gaps
        gaps = np.abs(arrival_times - arrival_times.T)

        return gaps, arrival_times[0]

    def _compute_fitness(
        self,
        paths: PathList,
    ) -> None:
        """
        Compute the actual fitness based on the paths provided in the chromosome.

        The fitness is the maximal variance accumulated by any of the 12 ships.

        Args:
            paths (PathList):
                Paths for each of the 12 ships.
        """

        variances = np.zeros((self.n_ships,))

        for ship, path in enumerate(paths):
            for src, tgt in zip(path[:-1], path[1:]):
                if src not in self.network.nodes or tgt not in self.network[src]:
                    break

                variances[ship] += self.network[src][tgt]["v"]

        self._fitness.append(variances.max() * self._scaling)

    def _evaluate(
        self,
        x: np.ndarray,
        log: bool = False,
        plot: bool = False,
    ) -> EvalRetType:
        """
        Computes the constraints and the fitness of the provided chromosome.

        1. Equality constraints:

            - Validity of the origin nodes (must be in the set of origin nodes for each ship)
            - Validity of the paths (the network is sparse, so not all edges exist)

        2. Inequality constraints:

            - Arrival time (must be within the window).

        3. Fitness:

            - The maximum variance accumulated by any of the 12 ships.

        Args:
            x (np.ndarray):
                An integer array representing the paths taken by the 12 ships.

            log (bool, optional):
                Log output flag. Defaults to False.

            plot (bool, optional):
                Plot flag. Defaults to False.

        Returns:
            EvalRetType:
                A tuple containing:
                    - A list containing the fitness and the constraints
                    - Optionally, a tuple containing a Matplotlib figure and axis
                        with plots of the mean and variance of the paths taken by
                        the ships through the wormhole network.
        """

        # Logging
        if log:
            logger.info("==[ Fitness evaluation ]==")

        # Reset the fitness evaluation attributes
        self._reset_fitness_attributes()

        # Convert the chromosome into a list of paths.
        paths = self._chromosome_to_paths(x)

        if log:
            for idx, path in enumerate(paths):
                logger.info(f"Ship {idx + 1:>2d} | path length: {len(path):>3d}")

        """
        1. Equality constraints.
        """

        # * Origin node equality constraints (dim: # of ships) * #
        origin_ec = np.ones((self.n_ships,), dtype=np.int32)

        # Entry point constraint violations
        for ship_idx, path in enumerate(paths):
            if path[0] in self.origins[ship_idx]:
                origin_ec[ship_idx] = 0

        if log:
            for idx, ec in enumerate(origin_ec):
                logger.info(
                    f"Ship {idx + 1:>2d} origin constraint: {('' if ec == 0 else 'not ')}satisfied"
                )

        # Update the equality constraints
        self._update_ecs(origin_ec.sum())

        # * Path constraints (# of ships) * #
        # Path constraint violations
        path_ec = np.ones((self.n_ships,), dtype=np.int32)

        for ship_idx, path in enumerate(paths):
            if nx.is_path(self.network, path):
                path_ec[ship_idx] = 0

        if log:
            for idx, ec in enumerate(path_ec):
                logger.info(
                    f"Ship {idx + 1:>2d} path constraint: {('' if ec == 0 else 'not ')}satisfied"
                )

        # Update the equality constraints
        self._update_ecs(path_ec.sum())

        """
        2. Inequality constraints.
        """

        # * Window constraints (# of ships) * #
        # Accumulate the path means and add them to the delays
        (gaps, _) = self._compute_arrival_gaps(paths)
        window_ic = gaps - self.window

        if log:
            for ship1 in range(self.n_ships):
                for ship2 in range(ship1 + 1, self.n_ships):
                    gap = gaps[ship1][ship2]
                    inout = "inside" if gap <= self.window else "outside"
                    logger.info(
                        f"Gap for ships {ship1 + 1:>2d} and {ship2 + 1:>2d}: {gap:>2.8f} ({inout} the window)"
                    )

        # Update the inequality constraints
        self._update_ics(window_ic.max())

        """
        3. Fitness
        """

        # Compute the fitness if the constraints are satisfied.
        self._compute_fitness(paths)

        if log:
            fitness = f"{self._fitness[0]:>2.8}" if len(self._fitness) > 0 else ""
            logger.info(f"Fitness | {fitness}")
            logger.info(
                f"The provided chromosome is {'' if self._all_constraints_satisfied else 'not '}a solution"
            )

        # Plot the paths if requested
        fig_ax = None
        if plot:
            if log:
                logger.info(f"Preparing plot")

            fig_ax = self._plot(paths)

        # Extend the fitness vector with the equality and inequality constraints
        self._compose_udp_retval()

        return (self._fitness, fig_ax)

    """
    ######################
        Public methods
    ######################
    """

    def fitness(
        self,
        x: np.ndarray,
    ) -> List[float]:
        """
        A wrapper for the fitness function called only for evaluation of the fitness.

        # * NOTE * #
            - The chromosome consists of the IDs of the origin node and the waypoint nodes along the path to the destination node.

            - *All node IDs are 1-based*. Evaluation of the chromosome for any ship stops
                at the first 0 or when the maximum number of jumps has been reached.

            - Paths must be provided sequentially for each ship:

                [path for the 1st ship, path for the 2nd ship, ..., path for the 12th ship]

            - It is assumed that the last jump is to the destination node, therefore it is not necessary to provide
                the ID of the destination node. Keep in mind is that a wormhole must exist between the
                last provided black hole and the destination node, otherwise the chromosome will not satisfy
                the second equality constraint.

        Args:
            x (np.ndarray):
                A chromosome.

        Returns:
            List[float]:
                A list containing:
                    - The fitness value(s) (cf. self.get_nobj())
                    - The equality constraints (cf. self.get_nec())
                    - The inequality constraints (cf. self.get_nic())
        """

        (retval, _) = self._evaluate(x)

        return retval

    def get_nobj(self) -> int:
        """
        The number of objectives.

        There is only one objective for this challenge:
        to minimise the maximum variance of the arrival time
        at the destination for any ship.

        Returns:
            int:
                The number of objectives.
        """
        return 1

    def get_nix(self) -> int:
        """
        The number of integer components of the chromosome.

        Each item in the chromosome is either an entry point or an exit point,
        so the chromosome contains *only* integers.

        There are 12 entry points and up to `jump_limit` exit points,
        so the chromosome is of length 12 * jump_limit.

        Returns:
            int:
                The number of integer components of the chromosome.
        """
        return 12 * self.jump_limit

    def get_nec(self) -> int:
        """
        The number of equality constraints.

        There are two equality constraints (cf. _evaluate() for details).

        Equality constraints:
            1. The first node in a ship's path must belong to that ship's predefined set of origin nodes.
            2. The provided chromosome must define valid paths through the network .

        Returns:
            int:
                The number of equality constraints.
        """
        return 2

    def get_nic(self) -> int:
        """
        The number of inequality constraints.

        There is one inequality constraint (cf. _evaluate() for details):

        Inequality constraints:
            1. The maximum difference of the mean arrival times of any pair of ships
            must be less than or equal to the window.

        Returns:
            int:
                The number of inequality constraints.
        """

        return 1

    def get_bounds(self) -> Tuple[np.ndarray]:
        """
        Bounds for all chromosome elements.

        Returns:
            Tuple[np.ndarray]:
                Bound constraint violations for each element in the chromosome.
        """

        lb = [0] * (self.n_ships * self.jump_limit)
        ub = [self.n_bh] * (self.n_ships * self.jump_limit)

        return (lb, ub)

    def pretty(
        self,
        x: np.ndarray,
    ) -> EvalRetType:
        """
        Fitness evaluation function with pretty printing.

        Args:
            x (np.ndarray):
                A chromosome.
        """

        (retval, _) = self._evaluate(x, log=True)

        return retval

    def plot(
        self,
        x: np.ndarray,
        log: Optional[bool] = False,
    ) -> Tuple[EvalRetType, FigAxes]:
        """
        Plot the paths taken by the ships to the destination.

        Args:
            x (np.ndarray):
                A chromosome.

            log (Optional[bool], optional):
                Print a verbose output. Defaults to False.

        Returns:
            FigAxes:
                Plot of the fitness and the window.
        """

        (retval, fig_ax) = self._evaluate(x, log=log, plot=True)

        return (retval, fig_ax)

    def example(
        self,
        filename: FSPath = "./data/spoc2/wormholes/example.npy",
    ) -> np.ndarray:
        """
        Return a minimal chromosome that satisfies the problem constraints.

        Args:
            filename (FSPath):
                Path to the NumPy array containing the example solution.

        Returns:
            np.ndarray:
                A valid chromosome.
        """

        filename = Path(filename)

        if not filename.exists():
            raise ValueError(f"The specified file '{filename}' does not exist!")

        example_chromosome = np.load(filename)
        return example_chromosome

    def convert_to_chromosome(
        self,
        x: PathList,
    ) -> np.ndarray:
        """
        Creates a valid chromosome from a list of paths.

        The paths might be shorter than the maximum allowable path length.
        This method completes the chromosome to the required length.

        Since the chromosome should contain only integers,
        the chromosome encoding can make it difficult to figure out
        where the path of one ship ends and that of another begins.
        To deal with this, we truncate the paths if they are
        too long or pad them with zeros if they are too short.
        This is the reason why all node IDs are 1-based.

        Args:
            x (PathList):
                List of paths through the network.

        Returns:
            np.ndarray:
                A valid chromosome.
        """

        # Create an empty chromosome
        chromosome = np.zeros((self.n_ships * self.jump_limit,), dtype=np.int32)

        if len(x) != 12:
            logger.info(f"Please provide a list of exactly 12 paths.")
            return chromosome

        for ship, path in enumerate(x):
            if len(path) > self.jump_limit:
                # Truncate paths that are too long
                path = path[: self.jump_limit]

            elif len(path) < self.jump_limit:
                # Extend paths that are too short
                path = np.concatenate(
                    (path, np.zeros((self.jump_limit - len(path))))
                ).astype(np.int32)

            # CRISPR the path into the chromosome
            chromosome[ship * self.jump_limit : (ship + 1) * self.jump_limit] = path

        return chromosome


udp = wormhole_traversal_udp()
