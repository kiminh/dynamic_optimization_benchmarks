from numpy.random import rand, exponential


class basic_vertex(object):
    """
    Represents a vertex to be matched in the system.
    Implements three methods:
    - match_value (value of matching to another vertex of the same type).
    important that a.match_value(b) == b.match_value(a)
    - unmatched_value: value of leaving unmatched. important that a.unmatched_value == a.b
    For now represents a very basic structure where assortative matching is optimal
    """

    def __init__(self, arr_time):
        if rand() < 0.5:
            self.type = "1"
        else:
            self.type = "2"  # two types "1" and "2" for now. TODO: change to attributes
        self.arrival_time = arr_time
        self.id = arr_time
        self.dep_time = arr_time + exponential(10)
        # For now, vertices are uniquely identified by when they arrive (only one arrival per time step)

    def match_value(self, other_vertex):
        if other_vertex == self:
            return 0  # self.unmatched_value()  # self match is the same as leaving unmatched.
        elif self.type == other_vertex.type:
            return 10
        else:
            return 1

    def unmatched_value(self):
        return 0

    def departure(self, time):
        if time >= self.dep_time:
            return True
        else:
            return False


class basic_vertex_generator(object):

    def __init__(self):
        self.name = "basic"
        self.data_dir = ""
        self.mode = ""

    def new_vertex(self, i):
        return basic_vertex(i)
