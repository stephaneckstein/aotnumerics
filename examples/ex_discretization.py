import numpy as np
from src.measures import Graph, trinomial, empirical_k_means_measure, brownian_motion_sample, get_marginals_multi
from src.mainfunctions import gurobi_bm
from time import time


T = 3  # includes time 0
# Markovian structure:
g = Graph(T)
for t in range(T - 1):
    g.addEdge(t, t + 1)
nu, supp_nu = trinomial()
nu_marg_full = get_marginals_multi(nu, supp_nu, list(range(T)), g, [])


def f_1(x, y):
    return np.sum(np.abs(x - y))


ss_each = 10
n_list = [20, 40, 75, 100, 200, 500, 1000, 1500, 2000, 3000, 5000, 7500, 10000]
values_bc_emp = np.zeros([len(n_list), ss_each])
values_bc_ademp = np.zeros([len(n_list), ss_each])
values_ot_emp = np.zeros([len(n_list), ss_each])
values_ot_ademp = np.zeros([len(n_list), ss_each])
times_bc_emp = np.zeros([len(n_list), ss_each])
times_bc_ademp = np.zeros([len(n_list), ss_each])
times_ot_emp = np.zeros([len(n_list), ss_each])
times_ot_ademp = np.zeros([len(n_list), ss_each])


for ind_n, n in enumerate(n_list):
    for ind_s in range(ss_each):
        bms = brownian_motion_sample(T - 1, n)
        mu_full = [bms, np.ones(len(bms)) / len(bms)]
        mu_ad_x, mu_ad_w = empirical_k_means_measure(bms, use_weights=1)
        mu_full_ad = [mu_ad_x, mu_ad_w]
        if n < 2000:
            t0 = time()
            val0, _ = gurobi_bm([mu_full, nu_marg_full], f=f_1, r_opti=1, causal=1, anticausal=1, outputflag=0)
            times_bc_emp[ind_n, ind_s] = time() - t0
            values_bc_emp[ind_n, ind_s] = val0
            print('Sample points:', n, 'AOT value (empirical):', val0, 'Time:', time() - t0)
        t0 = time()
        val0, _ = gurobi_bm([mu_full_ad, nu_marg_full], f=f_1, r_opti=1, causal=1, anticausal=1, outputflag=0)
        times_bc_ademp[ind_n, ind_s] = time() - t0
        values_bc_ademp[ind_n, ind_s] = val0
        print('Sample points:', n, 'AOT value (adapted empirical):', val0, 'Time:', time() - t0)
        t0 = time()
        val0, _ = gurobi_bm([mu_full, nu_marg_full], f=f_1, r_opti=1, causal=0, anticausal=0, outputflag=0)
        times_ot_emp[ind_n, ind_s] = time() - t0
        values_ot_emp[ind_n, ind_s] = val0
        print('Sample points:', n, 'OT value (empirical):', val0, 'Time:', time() - t0)
        t0 = time()
        val0, _ = gurobi_bm([mu_full_ad, nu_marg_full], f=f_1, r_opti=1, causal=0, anticausal=0, outputflag=0)
        times_ot_ademp[ind_n, ind_s] = time() - t0
        values_ot_ademp[ind_n, ind_s] = val0
        print('Sample points:', n, 'OT value (adapted empirical):', val0, 'Time:', time() - t0)
np.save('../data/values_bc_emp', values_bc_emp)
np.save('../data/values_bc_ademp', values_bc_ademp)
np.save('../data/values_ot_emp', values_ot_emp)
np.save('../data/values_ot_ademp', values_ot_ademp)
np.save('../data/times_bc_emp', times_bc_emp)
np.save('../data/times_bc_ademp', times_bc_ademp)
np.save('../data/times_ot_emp', times_ot_emp)
np.save('../data/times_ot_ademp', times_ot_ademp)

