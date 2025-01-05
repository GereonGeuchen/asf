import pandas as pd
import numpy as np
import math
import clingo

from asf.presolving.presolver import AbstractPresolver


# Python functions to handle custom operations
def insert_factory(ts):
    def insert(i, s, t):
        key = str(s)
        if key not in ts:
            ts[key] = []
        ts[key].append((i, t))
        return clingo.Number(1)

    return insert


def order_factory(ts):
    def order(s):
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
    def __init__(self, metadata, cores: int, cutoff: int):
        super().__init__(metadata)
        self.cores = cores
        self.cutoff = cutoff
        self.data_threshold = 300  # minimal number of instances to use
        self.data_fraction = 0.3  # fraction of instances to use

    def _fit(self, features: pd.DataFrame, performance: pd.DataFrame):
        ts = {}

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

        def clingo_callback(model):
            schedule_dict = {}
            for slice in model.symbols(shown=True):
                algo = self.metadata.algorithms[slice.arguments[1].number]
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
        return self.schedule
