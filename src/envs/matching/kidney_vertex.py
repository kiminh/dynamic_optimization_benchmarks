from numpy.random import choice, exponential, uniform
import pandas as pd


class unweighted_kidney_vertex(object):
    """
    Represents the case where the vertex is a patient-donor pair.
    Matching two pairs yields 1.
    """

    def __init__(self, compat_matrix, id, data_id, arr_time,
                 dep_rate=100, dep_mode="deterministic",
                 features=None):
        self.id = id
        self.data_id = data_id
        self.arr_time = arr_time
        self.compatibility_matrix = compat_matrix
        self.features = features
        if dep_mode == "deterministic":
            self.dep_time = self.arr_time + dep_rate
        elif dep_mode == "exponential":
            self.dep_time = arr_time + exponential(dep_rate)
        elif dep_mode == "uniform":
            self.dep_time = arr_time + uniform(0, 2*dep_rate)
        elif dep_mode == "heterogeneous_exp":
            if self.features.loc[data_id, "PRA"] > 80:
                self.dep_time = arr_time + exponential(2 * dep_rate)
            else:
                self.dep_time = arr_time + exponential(dep_rate)
        else:
            assert False

    def match_value(self, other_vertex):
        """
        The value is 1 if they are compatible, 0 otw.
        """
        return self.compatibility_matrix.ix[self.data_id, other_vertex.data_id]

    def unmatched_value(self):
        return 0

    def departure(self, time):
        if time >= self.dep_time:
            return True
        else:
            return False


class unweighted_kidney_vertex_generator(object):
    def __init__(self, directory, mode="random", dep_rate=100,
                 dep_mode="deterministic", iterations=10000):
        """
        directory: relative path to the data where the trips are stored.
        """
        self.data_dir = directory
        self.compatibility_matrix = pd.read_csv(self.data_dir + "compatibility_matrix.txt", sep='\t', header=None)
        self.features = pd.read_csv(self.data_dir + "pairs_pra.csv")
        self.mode = mode
        self.max_iterations = iterations
        self.order = choice(len(self.compatibility_matrix), iterations)
        self.name = "kidney_unweighted"
        self.total_dist_unmatched = 0
        self.departure_rate = dep_rate
        self.departure_mode = dep_mode
        assert self.mode == 'random' or self.max_iterations <= 1661

    def new_vertex(self, i):
        assert i <= self.max_iterations
        id = i
        arr_time = i
        if self.mode == "deterministic":
            data_id = i
        elif self.mode == "random":
            data_id = self.order[i]
        else:
            assert False
        return unweighted_kidney_vertex(self.compatibility_matrix, id, data_id, arr_time,
                                        dep_rate=self.departure_rate,
                                        dep_mode=self.departure_mode,
                                        features=self.features)
