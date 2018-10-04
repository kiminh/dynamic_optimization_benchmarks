import networkx as nx
# from src.algorithms.shadow_price_learner import shadow_price_learner
# from src.lp_mip.py import gurobi_max_weight_matching
class re_opt_matching(object):
    """
    Implements a re-optimization matching algorithm, with multiple
    Computes a max-weight matching of the current graph,
    and matches the vertices that are about to depart.
    """

    def __init__(self,
                 method="nx",
                 alpha=1,
                 select_match_mode='all',
                 update_weight_mode='none',
                 patience=0,
                 name='nan',
                 batch_size=1,
                 shadow_price=0
                 ):
        self.name=name
        self.method = method
        self.version = '0.5'
        self.alpha = alpha
        self.select_match_mode = select_match_mode
        self.update_weight_mode = update_weight_mode
        self.patience = patience
        self.batch_size = batch_size
        # if update_weight_mode == 'learned_shadow_prices':
        #     self.learner = shadow_price_learner(mode="PRA")

    def find_matching(self, state, prev_reward,  t):
        if self.name != 'batching' or t % self.batch_size == 0:
            self.update_weights(state, t, self.alpha, self.update_weight_mode)
            mate = self.find_max_weight_matching(state)
            return self.select_matches(mate, t, state, mode = self.select_match_mode,
                                       patience = self.patience)
        else:
            return []

    def final_step(self, state, t):
        mate = self.find_max_weight_matching(state)
        return self.select_matches(mate, t, state, 'all')

    def update_weights(self, state, t, alpha = 1.1, mode ='none'):
        """
        Changes
        """
        if mode == 'none':
            pass
        elif mode == 'departing':
            for (i,j) in state.edges():
                if i.departure(t) or j.departure(t):
                    state[i][j]["weight"] = alpha * state[i][j]["true_w"]
                else:
                    state[i][j]["weight"] = state[i][j]["true_w"]
        elif mode == 'waiting_time':
            for (i,j) in state.edges():
                if i.arr_time + self.patience <= t or j.arr_time + self.patience <= t:
                    state[i][j]["weight"] = alpha * state[i][j]["true_w"]
                else:
                    state[i][j]["weight"] = state[i][j]["true_w"]
        elif mode == 'mult_alpha_dep':
            for (i,j) in state.edges():
                t_i = max(i.dep_time - t, 0)
                t_j = max(j.dep_time - t, 0)
                state[i][j]["weight"] = alpha**(- t_i - t_j) * state[i][j]["true_w"]
        elif mode == 'mult_alpha_wait':
            for (i,j) in state.edges():
                t_i = max(i.arr_time + self.patience - t, 0)
                t_j = max(j.arr_time + self.patience - t, 0)
                state[i][j]["weight"] = alpha**(- t_i - t_j) * state[i][j]["true_w"]
        elif mode == 'shadow_price':
            for (i,j) in state.edges():
                state[i][j]["weight"] = max(state[i][j]["true_w"] - 2*shadow_price, 0)
        # elif mode == 'learned_shadow_prices':
        #     self.learner.train_step(state, prev_reward)
        #     for (i,j) in state.edges():
        #         state[i][j]["weight"] = max(state[i][j]["true_w"] - \
        #                                         self.learner.find_shadow_price(i) -\
        #                                         self.learner.find_shadow_price(j), 0)

    def find_max_weight_matching(self, state):
        mate = nx.algorithms.matching.max_weight_matching(state)
        # changes output to a dictionary.
        mate = {i:j for (i,j) in mate}
        return mate

    def select_matches(self, mate, t, state, mode='all', patience=0):
        if mode == 'all':
            return [(mate[k], k) for k in mate.keys() if
                       (mate[k].id <= k.id) and
                       (state[k][mate[k]]['weight'] > 0)
                       ]
        elif mode == 'departing':
            return [(mate[k], k) for k in mate.keys() if
                    (k.departure(t) or mate[k].departure(t)) and
                    (mate[k].id <= k.id) and
                    (state[k][mate[k]]['weight'] > 0)
                    ]
        elif mode == 'waiting_time':
            return [(mate[k], k) for k in mate.keys() if
                    (k.arr_time + patience <= t or \
                        mate[k].arr_time + patience <= t) and
                    (mate[k].id <= k.id) and
                    (state[k][mate[k]]['weight'] > 0)
                    ]
        elif mode == 'PRA_waiting':
            return [(mate[k], k) for k in mate.keys() if
                    (k.arr_time + (k.features.loc[k.data_id, "PRA"] > 80 + 1)*patience <= t \
                     or mate[k].arr_time + (mate[k].features.loc[mate[k].data_id, "PRA"] > 80 + 1)
                        *patience <= t) and
                    (mate[k].id <= k.id) and
                    (state[k][mate[k]]['weight'] > 0)
                    ]
        elif mode == 'dist_waiting':
            return [(mate[k], k) for k in mate.keys() if
                    (k.arr_time + k.features['dist'] * patience / 2.19 <= t \
                     or mate[k].arr_time + mate[k].features['dist'] * patience / 2.19 <= t) and
                    (mate[k].id <= k.id) and
                    (state[k][mate[k]]['weight'] > 0)
                    ]
