from src.sim import simulation
from generate_sim_plan import parameter_set, generate_sim_plan
import multiprocessing as mp
import time
import sys

def run_sim(p):
    sim = simulation(p)
    sim.run()

def run_multiple_sim(sim_plan):
    for p in sim_plan:
        run_sim(p)

def run_paralell_sim(sim_plan):
    with mp.Pool(processes=mp.cpu_count() - 1) as pool:
        pool.map(run_sim, sim_plan, chunksize=1)

if __name__ == '__main__':
    if len(sys.argv) == 4:
        T = int(sys.argv[1])
        alg = sys.argv[2]
        arr = sys.argv[3]
        p = parameter_set(T, alg, arr,
                          save=False,
                          batch=25,
                          patience=15,
                          alpha=1,
                          dep_rate=50,
                          r_seed = 2)
        run_multiple_sim([p])
    elif len(sys.argv) == 1:
        sys.setrecursionlimit(1500) # Needed to run nx.max_weight_matching on large graphs
        timer = time.time()
        sim_plan = generate_sim_plan(save=True)
        run_paralell_sim(sim_plan)
        # run_sim(sim_plan[0])
        print("Total time = {}".format(time.time() - timer))
