from numpy.random import permutation, exponential, uniform
import pandas as pd
import numpy as np

class taxi_vertex(object):
    """
    Represents the case where the vertex is an origin-destination trip.
    Matching two trips yields the value corresponding to the combined trip that visits both origins and both
    destinations in the right order.
    """

    def __init__(self, origin, dest, id, arr_time, patience=100, dep_mode="deterministic"):
        self.origin = origin  # a pair [lat, lng]
        self.dest = dest
        self.features = {'origin':origin, 'dest':dest, 'dist':self.euclidian_dist(origin, dest)}
        self.id = id
        self.arr_time = arr_time
        self.trip_length = self.euclidian_dist(self.origin, self.dest)
        if dep_mode == "deterministic":
            self.dep_time = self.arr_time + patience
        elif dep_mode == "exponential":
            self.dep_time = arr_time + exponential(patience)
        elif dep_mode == "uniform":
            self.dep_time = arr_time + uniform(0, 2*patience)
        elif dep_mode == "heterogeneous_exp":
            rate = patience * self.euclidian_dist(origin, dest) / 2.19
            # 2.19 is the average dist in our dataset.
            # This normalization is so that departure times are comparable to kidney simulations
            self.dep_time = arr_time + exponential(rate)
        else:
            assert False

    def euclidian_dist(self, a, b):
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def match_value(self, other_vertex):
        """
        The value is the difference between the euclidian combined trip length and the sum of euclidian trip lengths.
        Depends on a 'taxi_data' object that downloads and pre-processes the data.
        TODO: find better implementation
        """
        if self == other_vertex:
            return 0
        # self is a, other is b
        oa_ob = self.euclidian_dist(self.origin, other_vertex.origin)
        oa_db = self.euclidian_dist(self.origin, other_vertex.dest)
        ob_da = self.euclidian_dist(other_vertex.origin, self.dest)
        da_db = self.euclidian_dist(self.dest, other_vertex.dest)
        abab = oa_ob + ob_da + da_db
        abba = oa_ob + other_vertex.trip_length + da_db
        baba = oa_ob + oa_db + da_db
        baab = oa_ob + self.trip_length + da_db
        aabb = self.trip_length + other_vertex.trip_length  # no match
        match_value = aabb - min(abab, abba, baba, baab, aabb)
        assert match_value <= aabb / 2
        return match_value

    def unmatched_value(self):
        return 0

    def departure(self, time):
        if time >= self.dep_time:
            return True
        else:
            return False


class taxi_vertex_generator(object):
    def __init__(self, directory, mode="deterministic", dep_rate=100,
                 dep_mode="deterministic", shift_arrivals=0):
        """
        directory: relative path to the data where the trips are stored.
        """
        self.data_dir = directory
        self.df = pd.read_csv(directory)
        self.mode = mode
        self.order = permutation(range(len(self.df)))
        self.name = "taxi"
        self.total_dist_unmatched = 0
        self.departure_rate = dep_rate
        self.departure_mode = dep_mode
        self.shift_arrivals = shift_arrivals


    def new_vertex(self, i):
        arr_time = i
        if self.mode == "deterministic":
            id = i + self.shift_arrivals
        elif self.mode == "random":
            id = self.order[i + self.shift_arrivals]
        else:
            assert False
        origin = (float(self.df[['pX']].loc[id]), float(self.df[['pY']].loc[id]))
        dest = (float(self.df[['dX']].loc[id]), float(self.df[['dY']].loc[id]))
        self.total_dist_unmatched += float(self.df[['dist']].loc[i])
        return taxi_vertex(origin, dest, id, arr_time, patience=self.departure_rate, dep_mode=self.departure_mode)
