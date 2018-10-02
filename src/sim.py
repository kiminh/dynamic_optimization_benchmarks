from src.envs.matching.matching_env import matching_env
import time
import pandas as pd
from src.algorithms.re_opt import re_opt_matching
from src.algorithms.offline import offline_matching
from itertools import count
import os
from pathlib import Path

class simulation(object):
    """
    A full simulation
    """

    def __init__(self, p): #time_steps, algorithm, vertex_generator, verbose=0, save=False, seed=1234):
        self.time_steps = p.T

        self.algorithm = self.get_algorithm(p.alg_name, batch=p.batch,
                                       alpha=p.alpha, threshold=p.threshold,
                                       save=p.save, patience=p.patience,
                                       shadow_price=p.shadow_price)
        self.verbose = p.verbose
        self.save = p.save
        self.seed = p.r_seed
        self.p = p
        self.runtime = -1
        self.reward = 0
        # self.count = next(self._ids)
        self.process = os.getpid()
        self.sim_results_dir = "results/" + p.results_dir + "/" + str(self.process) + ".csv"
        s = str(p)
        print("Process {}, ".format(self.process) + s)

    def run(self):
        timer = time.time()
        env = matching_env(self.p)
        state = env.reset()
        action = self.algorithm.find_matching(state, 0)
        for t in range(self.time_steps):
            state, prev_reward = env.step(action)
            action = self.algorithm.find_matching(env, prev_reward, 0)
        final_match = self.algorithm.final_step(env, t)
        env.match(final_match)
        if self.offline:
            self.reward = self.algorithm.find_matching(env.offline_graph)
        else:
            self.reward = env.total_reward
        self.runtime = time.time() - timer
        if self.save:
            self.save_results(self.sim_results_dir)
        return self.reward

    def get_algorithm(self, alg, batch=1, alpha=1, threshold=0, save=False, patience=0, shadow_price=0):
        algorithms = {
            'greedy':re_opt_matching(name='greedy'),
            'batching':re_opt_matching(batch_size=batch, name='batching'),
            'd_re-opt':re_opt_matching(select_match_mode='departing',
                                       name='d_re-opt'),
            'd_alpha-re-opt':re_opt_matching(alpha=alpha,
                                             select_match_mode='departing',
                                             update_weight_mode='departing',
                                             name='d_alpha-re-opt'),
            're-opt':re_opt_matching(select_match_mode='waiting_time',
                                     name='re-opt',
                                     patience=patience),
            'alpha-re-opt':re_opt_matching(alpha=alpha,
                                           select_match_mode='waiting_time',
                                           update_weight_mode='waiting_time',
                                           name='alpha-re-opt',
                                           patience=patience),
            'offline':offline_matching(),
            'mult_alpha':re_opt_matching(alpha=alpha,
                                           select_match_mode='waiting_time',
                                           update_weight_mode='mult_alpha_wait',
                                           name='mult_alpha',
                                           patience=patience),
            'd_mult_alpha':re_opt_matching(alpha=alpha,
                                           select_match_mode='departing',
                                           update_weight_mode='mult_alpha_dep',
                                           name='d_mult_alpha'),
            'shadow_price':re_opt_matching(name='shadow_price',
                                           shadow_price=shadow_price),
            'PRA_waiting':re_opt_matching(select_match_mode='PRA_waiting',
                                          name='PRA_waiting',
                                          patience=patience),
            'learned_prices':re_opt_matching(select_match_mode='all',
                                             name='learned_prices',
                                             update_weight_mode='learned_shadow_prices'),
            'dist_waiting':re_opt_matching(select_match_mode='dist_waiting',
                                          name='dist_waiting',
                                          patience=patience)
        }
        return algorithms[alg]

    def save_results(self, sim_results_dir, mode="append"):
        if Path(sim_results_dir).is_file():
            self.df = pd.read_csv(sim_results_dir)
        else:
            print("couldn't find {}, creating file".format(sim_results_dir))
            self.df = pd.DataFrame()
        n, m = self.df.shape
        self.df.loc[n, "timestamp"] = str(pd.Timestamp.now()).replace(" ", "_")
        self.df.loc[n, "algorithm"] = self.algorithm.name
        self.df.loc[n, "algorithm_version"] = self.algorithm.version
        self.df.loc[n, "arrivals"] = self.vertex_generator.mode \
            if hasattr(self.vertex_generator, 'mode') else ""
        self.df.loc[n, "departure_mode"] = self.vertex_generator.departure_mode \
            if hasattr(self.vertex_generator, 'departure_mode') else ""
        self.df.loc[n, "departure_rate"] = self.vertex_generator.departure_rate \
            if hasattr(self.vertex_generator, 'departure_rate') else ""
        self.df.loc[n, "arr_data_shift"] = self.vertex_generator.shift_arrivals \
            if hasattr(self.vertex_generator, 'shift_arrivals') else 0
        self.df.loc[n, "graph_type"] = self.vertex_generator.name
        self.df.loc[n, "iterations"] = self.time_steps
        self.df.loc[n, "data_dir"] = self.vertex_generator.data_dir
        self.df.loc[n, "seed"] = self.seed
        self.df.loc[n, "runtime"] = round(self.runtime, 3)
        self.df.loc[n, "reward"] = round(self.reward, 3)
        if self.vertex_generator.name == "taxi":
            self.df.loc[n, "taxi_tot_dist"] = round(self.vertex_generator.total_dist_unmatched, 3)
            self.df.loc[n, "taxi_efficiency"] = round(1 - self.reward / self.vertex_generator.total_dist_unmatched, 3)
        if self.algorithm.name == "batching":
            self.df.loc[n, "batch_size"] = self.p.batch
        self.df.loc[n, "alpha"] = self.p.alpha
        self.df.loc[n, "patience"] = self.p.patience
        self.df.loc[n, "shadow_price"] = self.p.shadow_price
        self.df.to_csv(sim_results_dir, index=False)
