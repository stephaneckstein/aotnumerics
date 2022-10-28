import numpy as np
from src.mainfunctions import sinkhorn_causal_markov, sinkhorn_bicausal_markov, solve_dynamic, gurobi_bm
from src.measures import get_meas_for_sinkhorn, get_full_index_markov, get_start_next_indices, get_joint_prob, get_marginals_multi, Graph, rand_tree_pichler
from time import time

# Program to compute and save values for Table 1 of Eckstein & Pammer "Computational methods ..."

T = 2  # number of non-trivial time-steps (time 0 starting at 0 is not included)
UDRANGE = 100
N_BRANCH_LIST = [10, 25, 50]
EPS_LIST = [0.1, 0.01]
N_SAMPLE_RUNS = 10
SAVE = 1  # whether to save values (1) or not (0)
g = Graph(T + 1)  # indices of the graph are 0, 1, 2
# Markovian structure:
for t in range(T):
    g.addEdge(t, t + 1)


def cost_f_scalar_1(x, y):
    return np.abs(x-y)**2 / (4 * UDRANGE**2)


def cost_f_scalar_2(x, y):
    return np.sin(x*y) + np.abs(x-y)/UDRANGE


def cost_f_1(x, y):
    return np.abs(x[0]-y[0])**2/ (4 * UDRANGE**2)


def cost_f_2(x, y):
    return np.sin(x[0]*y[0]) + np.abs(x[0]-y[0])/UDRANGE


def f_1(x, y):
    return np.sum(np.abs(x-y)**2) / (4 * UDRANGE**2)


def f_2(x, y):
    t_max = len(x)
    return np.sum([np.sin(x[t]*y[t]) + np.abs(x[t]-y[t])/UDRANGE for t in range(t_max)])


def get_val_product(mu, supp_mu, nu, supp_nu, cost_scal, g, t_max):
    mu_mult = get_marginals_multi(mu, supp_mu, list(range(t_max)), g, given_margs={})
    nu_mult = get_marginals_multi(nu, supp_nu, list(range(t_max)), g, given_margs={})
    tot_cost = 0
    for i in range(len(mu_mult[1])):
        for j in range(len(nu_mult[1])):
            x = mu_mult[0][i, :]
            y = nu_mult[0][j, :]
            dh = 0
            for t in range(t_max):
                dh += cost_scal(x[t], y[t])
            tot_cost += dh * nu_mult[1][j] * mu_mult[1][i]
    return tot_cost


for N_BRANCH in N_BRANCH_LIST:
    for s_ind_h in range(N_SAMPLE_RUNS):
        str_ident = '../data/'+str(T)+'_'+str(UDRANGE)+'_'+str(N_BRANCH)+'_'+str(s_ind_h)
        np.random.seed(s_ind_h)  # for reproducability
        mu, supp_mu = rand_tree_pichler(T, num_branch=tuple([N_BRANCH] * T), udrange=UDRANGE)
        nu, supp_nu = rand_tree_pichler(T, num_branch=tuple([N_BRANCH] * T), udrange=UDRANGE)

        # Getting product measure values for relative errors
        val_prod_1 = get_val_product(mu, supp_mu, nu, supp_nu, cost_f_scalar_1, g, T+1)
        val_prod_2 = get_val_product(mu, supp_mu, nu, supp_nu, cost_f_scalar_2, g, T+1)
        if SAVE:
            np.save(str_ident + '_valprod_1', val_prod_1)
            np.save(str_ident + '_valprod_2', val_prod_2)
        print('Values for product measure:', val_prod_1, val_prod_2)

        # LP
        if N_BRANCH <= 25:
            mu_marg_full = get_marginals_multi(mu, supp_mu, list(range(T+1)), g, [])
            nu_marg_full = get_marginals_multi(nu, supp_nu, list(range(T+1)), g, [])

            t0 = time()
            LP_bc_v1, _ = gurobi_bm([mu_marg_full, nu_marg_full], f=f_1, r_opti=1, causal=1, anticausal=1, outputflag=0)
            t_lp_bc_1 = time()-t0
            t0 = time()
            LP_bc_v2, _ = gurobi_bm([mu_marg_full, nu_marg_full], f=f_2, r_opti=1, causal=1, anticausal=1, outputflag=0)
            t_lp_bc_2 = time() - t0
            print('Values for LP bicausal', LP_bc_v1, LP_bc_v2)
            print('Times for LP bicausal', t_lp_bc_1, t_lp_bc_2)
            if SAVE:
                np.save(str_ident + '_val_lp_bc_1', LP_bc_v1)
                np.save(str_ident + '_val_lp_bc_2', LP_bc_v2)
                np.save(str_ident + '_time_lp_bc_1', t_lp_bc_1)
                np.save(str_ident + '_time_lp_bc_2', t_lp_bc_2)

            t0 = time()
            LP_c_v1, _ = gurobi_bm([mu_marg_full, nu_marg_full], f=f_1, r_opti=1, causal=1, anticausal=0, outputflag=0)
            t_lp_c_1 = time()-t0
            t0 = time()
            LP_c_v2, _ = gurobi_bm([mu_marg_full, nu_marg_full], f=f_2, r_opti=1, causal=1, anticausal=0, outputflag=0)
            t_lp_c_2 = time() - t0
            print('Values for LP causal', LP_c_v1, LP_c_v2)
            print('Times for LP causal', t_lp_c_1, t_lp_c_2)
            if SAVE:
                np.save(str_ident + '_val_lp_c_1', LP_c_v1)
                np.save(str_ident + '_val_lp_c_2', LP_c_v2)
                np.save(str_ident + '_time_lp_c_1', t_lp_c_1)
                np.save(str_ident + '_time_lp_c_2', t_lp_c_2)


        # Backward induction
        cost_funs_1 = [[[t], cost_f_1] for t in range(T + 1)]
        cost_funs_2 = [[[t], cost_f_2] for t in range(T + 1)]

        t0 = time()
        BW_v1, _ = solve_dynamic(cost_funs_1, mu, nu, supp_mu, supp_nu, g, outputflag=0)
        BW_v1 = BW_v1[0]
        t_bw_bc_1 = time()-t0
        t0 = time()
        BW_v2, _ = solve_dynamic(cost_funs_2, mu, nu, supp_mu, supp_nu, g, outputflag=0)
        BW_v2 = BW_v2[0]
        t_bw_bc_2 = time()-t0

        print('Values for Backward induction bicausal', BW_v1, BW_v2)
        print('Times for Backward induction bicausal', t_bw_bc_1, t_bw_bc_2)
        if SAVE:
            np.save(str_ident + '_val_bw_bc_1', BW_v1)
            np.save(str_ident + '_val_bw_bc_2', BW_v2)
            np.save(str_ident + '_time_bw_bc_1', t_bw_bc_1)
            np.save(str_ident + '_time_bw_bc_2', t_bw_bc_2)

        t0 = time()
        BWs_v1, _ = solve_dynamic(cost_funs_1, mu, nu, supp_mu, supp_nu, g, outputflag=0, method='sinkhorn')
        BWs_v1 = BWs_v1[0]
        t_bws_bc_1 = time() - t0
        t0 = time()
        BWs_v2, _ = solve_dynamic(cost_funs_2, mu, nu, supp_mu, supp_nu, g, outputflag=0, method='sinkhorn')
        BWs_v2 = BWs_v2[0]
        t_bws_bc_2 = time() - t0

        print('Values for Backward induction (Sinkhorn) bicausal', BWs_v1, BWs_v2)
        print('Times for Backward induction (Sinkhorn) bicausal', t_bws_bc_1, t_bws_bc_2)
        if SAVE:
            np.save(str_ident + '_val_bws_bc_1', BWs_v1)
            np.save(str_ident + '_val_bws_bc_2', BWs_v2)
            np.save(str_ident + '_time_bws_bc_1', t_bws_bc_1)
            np.save(str_ident + '_time_bws_bc_2', t_bws_bc_2)

        # Sinkhorn
        for EPS in EPS_LIST:
            str_ident_eps = str_ident + '_' + str(EPS)
            x_list, mu_list = get_meas_for_sinkhorn(mu, supp_mu, T + 1)
            y_list, nu_list = get_meas_for_sinkhorn(nu, supp_nu, T + 1)
            ind_tot = get_full_index_markov(nu_list)
            ind_next_l = get_start_next_indices(ind_tot)
            nu_joint_prob = get_joint_prob(nu_list, ind_tot, T - 1)
            cost_mats = []
            for t in range(T + 1):
                cmh = np.zeros([len(x_list[t]), len(y_list[t])], dtype=np.float64)
                for i in range(len(x_list[t])):
                    for j in range(len(y_list[t])):
                        cmh[i, j] = np.exp(-1 / EPS * cost_f_scalar_1(x_list[t][i], y_list[t][j]))
                cost_mats.append(cmh)
            cost_mats_2 = []
            for t in range(T + 1):
                cmh_2 = np.zeros([len(x_list[t]), len(y_list[t])], dtype=np.float64)
                for i in range(len(x_list[t])):
                    for j in range(len(y_list[t])):
                        cmh_2[i, j] = np.exp(-1 / EPS * cost_f_scalar_2(x_list[t][i], y_list[t][j]))
                cost_mats_2.append(cmh_2)

            n_list = [len(x_list[i]) for i in range(T + 1)]
            m_list = [len(y_list[i]) for i in range(T + 1)]
            t0 = time()
            vs3 = sinkhorn_causal_markov(mu_list, nu_list, cost_mats, n_list, m_list, ind_tot, ind_next_l, nu_joint_prob, eps_stop=10**-4)
            t_sink_c_1 = time()-t0

            t0 = time()
            vs3_2 = sinkhorn_causal_markov(mu_list, nu_list, cost_mats_2, n_list, m_list, ind_tot, ind_next_l, nu_joint_prob, eps_stop=10**-4, reshape=0)
            t_sink_c_2 = time()-t0
            sink_c_v1 = vs3 * EPS
            sink_c_v2 = vs3_2 * EPS

            t0 = time()
            val_sink = sinkhorn_bicausal_markov(mu_list, nu_list, cost_mats, n_list, m_list, eps_stop=10**-4, reshape=0)
            t_sink_bc_1 = time()-t0
            t0 = time()
            val_sink_2 = sinkhorn_bicausal_markov(mu_list, nu_list, cost_mats_2, n_list, m_list, eps_stop=10**-4, reshape=0)
            t_sink_bc_2 = time()-t0

            sink_bc_v1 = val_sink * EPS
            sink_bc_v2 = val_sink_2 * EPS

            print('Values for Sinkhorn (causal), EPS = ' + str(EPS), sink_c_v1, sink_c_v2)
            print('Times for Sinkhorn (causal), EPS = ' + str(EPS), t_sink_c_1, t_sink_c_2)
            if SAVE:
                np.save(str_ident_eps + '_val_sink_c_1', sink_c_v1)
                np.save(str_ident_eps + '_val_sink_c_2', sink_c_v2)
                np.save(str_ident_eps + '_time_sink_c_1', t_sink_c_1)
                np.save(str_ident_eps + '_time_sink_c_2', t_sink_c_2)

            print('Values for Sinkhorn (bicausal), EPS = ' + str(EPS), sink_bc_v1, sink_bc_v2)
            print('Times for Sinkhorn (bicausal), EPS = ' + str(EPS), t_sink_bc_1, t_sink_bc_2)
            if SAVE:
                np.save(str_ident_eps + '_val_sink_bc_1', sink_bc_v1)
                np.save(str_ident_eps + '_val_sink_bc_2', sink_bc_v2)
                np.save(str_ident_eps + '_time_sink_bc_1', t_sink_bc_1)
                np.save(str_ident_eps + '_time_sink_bc_2', t_sink_bc_2)



