import networkx as nx
# from src.lp_mip import gurobi_max_weight_matching
import pandas as pd
import numpy as np
from time import time
import os
import pickle
import json


class offline_matching(object):
    """
    Implements the offline matching algorithm: does not match until the very end,
    returns a max-weight matching of the final graph.
    """

    def __init__(self, name='offline', method="nx", save=False):
        """
        method can be "nx" --> use the built-in max_weight_matching algorithm
        or "gurobi" --> uses the MIP formulation
        """
        self.name = name
        self.method = method
        self.version = '0.0'
        self.save = save

    def find_matching(self, offline_graph):
        if self.method == "nx":
            mate = nx.algorithms.matching.max_weight_matching(offline_graph)
            # change a dict into a list of pairs, remove duplicates.
            return np.sum([offline_graph[k][mate[k]]['true_w'] for k in mate.keys() if (mate[k].id <= k.id)])
        elif self.method == "gurobi":
            pairs, value = gurobi_max_weight_matching(offline_graph, mode="mip")
            return value


    def save_results(self, ts, runtime, sim, mode="append"):
        self.df = pd.read_csv("data/taxi/shadow_prices/log.csv")
        n, m = self.df.shape
        self.df.loc[n, "file_name"] = "{}.csv".format(ts)
        self.df.loc[n, "algorithm"] = self.name
        self.df.loc[n, "opt_method"] = self.method
        self.df.loc[n, "departure_mode"] = sim.vertex_generator.departure_mode
        self.df.loc[n, "departure_rate"] = sim.vertex_generator.departure_rate
        self.df.loc[n, "arr_data_shift"] = sim.vertex_generator.shift_arrivals
        self.df.loc[n, "graph_type"] = sim.vertex_generator.name
        self.df.loc[n, "iterations"] = sim.time_steps
        self.df.loc[n, "seed"] = sim.seed
        self.df.loc[n, "data_dir"] = sim.vertex_generator.data_dir
        self.df.loc[n, "runtime"] = runtime
        self.df.to_csv("data/taxi/shadow_prices/log.csv", index=False)
