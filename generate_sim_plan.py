from numpy.random import seed, rand
import numpy as np

class parameter_set(object):

    def __init__(self, T, alg, arr, batch=1, alpha=1, offset=0,
                 save=True, r_seed=1, threshold=0,
                 dep_rate=20,
                 dep_mode='exponential',
                 patience=0,
                 verbose=0,
                 results_dir = '',
                 shadow_price=0,
                 run_offline=False):
        self.T = T
        self.alg_name = alg
        self.vertex_generator_name = arr
        self.batch = batch
        self.alpha = alpha
        self.offset = offset
        self.save = save
        self.r_seed = r_seed
        seed(r_seed)
        self.threshold = threshold
        self.dep_rate = dep_rate
        self.dep_mode=dep_mode
        self.patience = patience
        self.verbose = 0
        self.results_dir = results_dir
        self.shadow_price = shadow_price
        self.run_offline = run_offline

    def __str__(self):
        s = "alg={}, T={}, data={}, seed={}, dep_rate={}, dep={}".format(self.alg_name, self.T,
            self.vertex_generator_name, self.r_seed, self.dep_rate,
            self.dep_mode)
        if self.alg_name in ['greedy', 'd_re-opt', 'offline']:
            return s
        elif self.alg_name == 'batching':
            return s + ", batch={}".format(self.batch)
        elif self.alg_name in ['d_alpha-re-opt', 'd_mult_alpha']:
            return s + ", alpha={}".format(self.alpha)
        elif self.alg_name == 're-opt':
            return s + ", patience={}".format(self.patience)
        elif self.alg_name in ['alpha-re-opt', 'mult_alpha']:
            return s + ", patience={}, alpha={}".format(self.patience, self.alpha)
        elif self.alg_name == 'shadow_price':
            return s + ", shadow_price={}".format(self.shadow_price)
        else:
            return s

def generate_sim_plan(save):
    """
    Input: Boolean:``save`` whether to save the results of the simulation.
    Output: a list of parameters.

    Parameters:
    - T: number of time steps to run the simulation for
    - random seed: r_seed. Is used to determine the directory in which to save the results.
    """
    T = 20
    sim_plan = []
    for r_seed in [32]:
        sim_results_dir="results" + str(r_seed)
        for dep_rate in [50]:#[150, 125, 100, 75, 50]:
            for dep_mode in ['uniform', 'deterministic', 'exponential']:
                for arr in ['taxi']:#, 'kidney_unweighted']:
                    # for dep_mode in ['uniform']:
                    if dep_mode == 'deterministic':
                        batch_array = [dep_rate]
                    elif arr == 'taxi':
                        batch_array = (dep_rate / np.array([10, 7, 5, 3, 2, 1.5, 1])).round()
                    elif arr == 'kidney_unweighted':
                        batch_array = (dep_rate / np.array([dep_rate, dep_rate/2, 20, 15, 10, 7, 5])).round()
                    # for alg in ['offline']:
                    for alg in ['greedy', 'd_re-opt']:
                        p = parameter_set(T, alg, arr, dep_rate=dep_rate,
                                          r_seed = r_seed,
                                          dep_mode = dep_mode,
                                          save=save,
                                          results_dir=sim_results_dir)
                        sim_plan.append(p)
                    for batch in batch_array:
                        alg = 'batching'
                        p = parameter_set(T, alg, arr, dep_rate=dep_rate,
                                          r_seed = r_seed,
                                          dep_mode = dep_mode,
                                          batch=batch,
                                          save=save,
                                          results_dir=sim_results_dir)
                        sim_plan.append(p)
                        alg = 're-opt'
                        p = parameter_set(T, alg, arr, dep_rate=dep_rate,
                                          r_seed = r_seed,
                                          dep_mode = dep_mode,
                                          patience=batch,
         # using batch_array instead of patience_array which would be the same
                                          save=save,
                                          results_dir=sim_results_dir)
                        sim_plan.append(p)
                    for alpha in [1.05, 1.1, 1.25, 1.5]:
                        p = parameter_set(T, 'd_alpha-re-opt',
                                          arr,
                                          dep_rate=dep_rate,
                                          r_seed = r_seed,
                                          dep_mode = dep_mode,
                                          alpha=alpha,
                                          save=save,
                                          results_dir=sim_results_dir)
                        sim_plan.append(p)
                        p = parameter_set(T, 'd_mult_alpha',
                                          arr,
                                          dep_rate=dep_rate,
                                          r_seed = r_seed,
                                          dep_mode = dep_mode,
                                          alpha= 1 + (alpha - 1) / 2 ,
                                          save=save,
                                          results_dir=sim_results_dir)
                        sim_plan.append(p)
                        for patience in batch_array:
                            alg = 'alpha-re-opt'
                            p = parameter_set(T, alg, arr, dep_rate=dep_rate,
                                              r_seed = r_seed,
                                              alpha=alpha,
                                              dep_mode = dep_mode,
                                              save=save,
                                              results_dir=sim_results_dir,
                                              patience=patience)
                            sim_plan.append(p)
                            p = parameter_set(T, 'mult_alpha',
                                              arr,
                                              dep_rate=dep_rate,
                                              r_seed = r_seed,
                                              dep_mode = dep_mode,
                                              alpha= 1 + (alpha - 1) / 2 ,
                                              save=save,
                                              results_dir=sim_results_dir,
                                              patience = patience)
                            sim_plan.append(p)
                    # for shadow_price in [0.1, 0.3, 0.5, 0.7, 0.8, 0.9]:
                    #     alg = 'shadow_price'
                    #     p = parameter_set(T, alg, arr, dep_rate=dep_rate,
                    #                       r_seed = r_seed,
                    #                       dep_mode = dep_mode,
                    #                       save=save,
                    #                       shadow_price=shadow_price,
                    #                       results_dir=sim_results_dir)
                    #     sim_plan.append(p)
    print("Created sim plan with {} simulations".format(len(sim_plan)))
    return sim_plan
