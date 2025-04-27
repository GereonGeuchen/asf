import pandas as pd
import numpy as np
import math
import clingo

from asf.presolving.presolver import AbstractPresolver


# Python functions to handle custom operations
def insert_factory(ts: dict) -> callable:
    """
    Creates an `insert` function that adds a tuple (i, t) to the list associated with a key in the dictionary `ts`.

    Args:
        ts (dict): A dictionary to store tuples of (i, t) indexed by string keys.

    Returns:
        callable: A function that inserts a tuple into the dictionary.
    """

    def insert(i: int, s: str, t: int) -> clingo.Number:
        """
        Inserts a tuple (i, t) into the dictionary `ts` under the key `s`.

        Args:
            i (int): An identifier.
            s (str): A string key.
            t (int): A value to be stored.

        Returns:
            clingo.Number: Always returns clingo.Number(1).
        """
        key = str(s)
        if key not in ts:
            ts[key] = []
        ts[key].append((i, t))
        return clingo.Number(1)

    return insert


def order_factory(ts: dict) -> callable:
    """
    Creates an `order` function that sorts and processes tuples in the dictionary `ts`.

    Args:
        ts (dict): A dictionary containing lists of tuples to be sorted.

    Returns:
        callable: A function that orders tuples and generates a list of clingo.Function objects.
    """

    def order(s: str) -> list[clingo.Function]:
        """
        Orders tuples in the dictionary `ts` under the key `s` and generates clingo.Function objects.

        Args:
            s (str): A string key.

        Returns:
            list[clingo.Function]: A list of clingo.Function objects representing ordered pairs.
        """
        key = str(s)
        if key not in ts:
            ts[key] = []
        ts[key].sort(key=lambda x: int(x[1]))
        p = None
        r = []
        for i, v in ts[key]:
            if p is not None:
                r.append(clingo.Function("", [p, i]))
            p = i
        return r

    return order


class Aspeed(AbstractPresolver):
    """
    A presolver class that uses Answer Set Programming (ASP) to compute a schedule for solving instances.

    Attributes:
        cores (int): Number of CPU cores to use.
        cutoff (int): Time limit for solving.
        data_threshold (int): Minimum number of instances to use.
        data_fraction (float): Fraction of instances to use.
        schedule (list): Computed schedule of algorithms and their budgets.
    """

    def __init__(self, metadata: dict, cores: int, cutoff: int) -> None:
        """
        Initializes the Aspeed presolver.

        Args:
            metadata (dict): Metadata for the presolver.
            cores (int): Number of CPU cores to use.
            cutoff (int): Time limit for solving.
        """
        super().__init__(metadata)
        self.cores = cores
        self.cutoff = cutoff
        self.data_threshold = 300  # minimal number of instances to use
        self.data_fraction = 0.3  # fraction of instances to use
        self.schedule: list[tuple[str, float]] = []

    def _fit(self, features: pd.DataFrame, performance: pd.DataFrame) -> None:
        """
        Fits the presolver to the given features and performance data.

        Args:
            features (pd.DataFrame): A DataFrame containing feature data.
            performance (pd.DataFrame): A DataFrame containing performance data.
        """
        ts: dict = {}

        # Create factories for the functions using the local `ts`
        insert = insert_factory(ts)
        order = order_factory(ts)

        # ASP program with dynamic number of cores
        asp_program = """
        #const cores={cores}.

        solver(S) :- time(_,S,_).
        time(S,T) :- time(_,S,T).
        unit(1..cores).

        insert(@insert(I,S,T)) :- time(I,S,T).
        order(I,K,S) :- insert(_), solver(S), (I,K) = @order(S).

        {{ slice(U,S,T) : time(S,T), T <= K, unit(U) }} 1 :- 
        solver(S), kappa(K).
        slice(S,T) :- slice(_,S,T).

        :- not #sum {{ T,S : slice(U,S,T) }} K, kappa(K), unit(U).

        solved(I,S) :- slice(S,T), time(I,S,T).
        solved(I,S) :- solved(J,S), order(I,J,S).
        solved(I)   :- solved(I,_).

        #maximize {{ 1@2,I: solved(I) }}.  
        #minimize {{ T*T@1,S : slice(S,T)}}.

        #show slice/3.
        """

        # Create a Clingo Control object with the specified number of threads
        ctl = clingo.Control(
            arguments=[f"--parallel-mode={self.cores}", f"--time-limit={self.cutoff}"]
        )

        # Register external Python functions
        ctl.register_external("insert", insert)
        ctl.register_external("order", order)

        # Load the ASP program
        ctl.add("base", [], asp_program)

        # if the instance set is too large, we subsample it
        if performance.shape[0] > self.data_threshold:
            random_indx = np.random.choice(
                range(performance.shape[0]),
                size=min(
                    performance.shape[0],
                    max(
                        int(performance.shape[0] * self.data_fraction),
                        self.data_threshold,
                    ),
                ),
                replace=True,
            )
            performance = performance[random_indx, :]

        times = [
            "time(i%d, %d, %d)." % (i, j, max(1, math.ceil(performance[i, j])))
            for i in range(performance.shape[0])
            for j in range(performance.shape[1])
        ]

        kappa = "kappa(%d)." % (self.presolver_cutoff)

        data_in = " ".join(times) + " " + kappa

        # Ground the logic program
        ctl.ground(data_in)

        def clingo_callback(model: clingo.Model) -> bool:
            """
            Callback function to process the Clingo model.

            Args:
                model (clingo.Model): The Clingo model.

            Returns:
                bool: Always returns False to stop after the first model.
            """
            schedule_dict = {}
            for slice in model.symbols(shown=True):
                algo = self.algorithms[slice.arguments[1].number]
                budget = slice.arguments[2].number
                schedule_dict[algo] = budget
                self.schedule = sorted(schedule_dict.items(), key=lambda x: x[1])
            return False

        # Solve the logic program
        with ctl.solve(yield_=False, on_model=clingo_callback) as result:
            if result.satisfiable:
                assert self.schedule is not None
            else:
                self.schedule = []

    def _predict(self) -> dict[str, list[tuple[str, float]]]:
        """
        Predicts the schedule based on the fitted model.

        Returns:
            dict[str, list[tuple[str, float]]]: A dictionary containing the schedule.
        """
        return self.schedule
