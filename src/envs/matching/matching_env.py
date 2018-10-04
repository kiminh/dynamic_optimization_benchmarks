import networkx as nx
from src.envs.matching.vertex import basic_vertex_generator
from src.envs.matching.taxi_vertex import taxi_vertex_generator
from src.envs.matching.kidney_vertex import unweighted_kidney_vertex_generator

class matching_env(object):
    """
    Implements a simple dynamic matching environment,
    where arrivals and departures are generated stochastically.
    Assumes that departing unmatched yields non-negative value.
    This allows us to remove negative-weight edges and sparsify the graph.
    """
    def __init__(self, p):
        self.vertex_generator = self.get_vertex_generator(p.vertex_generator_name,
                                                    offset=p.offset,
                                                    r_seed=p.r_seed,
                                                    iterations=p.T,
                                                    dep_rate=p.dep_rate,
                                                    dep_mode=p.dep_mode)
        self.reset()

    def reset(self):
        self.state = nx.Graph() # The state is a networkx graph object
        self.offline_graph = nx.Graph()
        self.present_vertices = []  # Keeps track of the offline graph.
        self.total_reward = 0
        self.last_reward = 0
        self.time = 0  # time is discrete.
        arriving_vertex = self.vertex_generator.new_vertex(self.time)
        self.arrival(arriving_vertex)
        return self.state

    def step(self, action):
        self.time += 1
        matched_vertices = action
        reward = self.match(matched_vertices)
        self.departures(self.time)
        arriving_vertex = self.vertex_generator.new_vertex(self.time)
        self.arrival(arriving_vertex)
        return self.state, reward

    def arrival(self, new_vertex):
        """
        Arrival of a new vertex to the system
        Authorizes self loops, it is up to the vertex class to declare 0 weights for self loops
        """
        # update state space
        self.state.add_node(new_vertex)
        for v in self.state.nodes():
            match_weight = new_vertex.match_value(v)
            if match_weight > 0 and v != new_vertex:
                self.state.add_edge(new_vertex, v, weight=match_weight, true_w=match_weight)
        # Update offline graph
        self.offline_graph.add_node(new_vertex)
        self.present_vertices.append(new_vertex)
        for v in self.present_vertices:
            match_weight = new_vertex.match_value(v)
            self.offline_graph.add_edge(new_vertex, v, weight=match_weight, true_w=match_weight)

    def departures(self, time):
        """
        Departure of all vertices that have been waiting too long.
        """
        reward = 0
        to_remove = []
        for v in self.state.nodes():
            if v.departure(time):
                to_remove.append(v)
                reward += v.unmatched_value()
        self.state.remove_nodes_from(to_remove)
        self.total_reward += reward
        self.last_reward += reward
        self.present_vertices[:] = [v for v in self.present_vertices if not v.departure(time)]
        return reward

    def match(self, matched_vertices):
        """
        Removes the matched_vertices, and computes the reward.
        Input: list of vertex pairs
        """
        reward = 0
        for (v1, v2) in matched_vertices:
            reward += self.state[v1][v2]['true_w']
            if v1 == v2:
                self.state.remove_node(v1)
            else:
                self.state.remove_node(v1)
                self.state.remove_node(v2)
        self.last_reward += reward
        self.total_reward += reward
        return reward

    def get_vertex_generator(self, arr, offset=0, r_seed=1, iterations=10000, dep_rate=10,
                             dep_mode='deterministic'):
        if arr == 'basic':
            return basic_vertex_generator()
        elif arr == 'taxi':
            return taxi_vertex_generator("data/taxi/rides.csv",
                                         mode="deterministic",
                                         dep_mode=dep_mode,
                                         shift_arrivals=offset,
                                         dep_rate=dep_rate)
        elif arr == 'kidney_unweighted':
            return unweighted_kidney_vertex_generator('data/kidney/',
                                                      mode='random',
                                                      dep_mode=dep_mode,
                                                      dep_rate=dep_rate,
                                                      iterations=iterations)
        else:
            assert False, "Arrival type {} not supported".format(arr)
