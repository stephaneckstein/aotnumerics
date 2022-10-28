import numpy as np

T = 2  # number of non-trivial time-steps (time 0 starting at 0 is not included)
UDRANGE = 100
N_BRANCH_LIST = [10, 25, 50]
EPS_LIST = [0.1, 0.01]
N_SAMPLE_RUNS = 10

run_times = np.zeros([N_SAMPLE_RUNS, len(N_BRANCH_LIST), 2, 2, 5])  # first 2 is cost, second 2 is c/bc, 5 methods
# method 1: LP; method 2: BI; method 3: BI(Sink), method 4: Sink(eps 0.1) method 5: Sink(eps 0.01)
values = np.zeros([N_SAMPLE_RUNS, len(N_BRANCH_LIST), 2, 2, 5])
prod_values = np.zeros([N_SAMPLE_RUNS, len(N_BRANCH_LIST), 2])
relative_errors = np.zeros([N_SAMPLE_RUNS, len(N_BRANCH_LIST), 2, 2, 5])
for i_nb, N_BRANCH in enumerate(N_BRANCH_LIST):
    for s_ind_h in range(N_SAMPLE_RUNS):
        str_ident = '../data/'+str(T)+'_'+str(UDRANGE)+'_'+str(N_BRANCH)+'_'+str(s_ind_h)
        val_prod_1 = np.load(str_ident + '_valprod_1.npy')
        val_prod_2 = np.load(str_ident + '_valprod_2.npy')
        prod_values[s_ind_h, i_nb, 0] = val_prod_1
        prod_values[s_ind_h, i_nb, 1] = val_prod_2

        # LP
        if N_BRANCH <= 25:
            LP_bc_v1 = np.load(str_ident + '_val_lp_bc_1.npy')
            LP_bc_v2 = np.load(str_ident + '_val_lp_bc_2.npy')
            t_lp_bc_1 = np.load(str_ident + '_time_lp_bc_1.npy')
            t_lp_bc_2 = np.load(str_ident + '_time_lp_bc_2.npy')
            values[s_ind_h, i_nb, 0, 1, 0] = LP_bc_v1
            values[s_ind_h, i_nb, 1, 1, 0] = LP_bc_v2
            run_times[s_ind_h, i_nb, 0, 1, 0] = t_lp_bc_1
            run_times[s_ind_h, i_nb, 1, 1, 0] = t_lp_bc_2

            LP_c_v1 = np.load(str_ident + '_val_lp_c_1.npy')
            LP_c_v2 = np.load(str_ident + '_val_lp_c_2.npy')
            t_lp_c_1 = np.load(str_ident + '_time_lp_c_1.npy')
            t_lp_c_2 = np.load(str_ident + '_time_lp_c_2.npy')
            values[s_ind_h, i_nb, 0, 0, 0] = LP_c_v1
            values[s_ind_h, i_nb, 1, 0, 0] = LP_c_v2
            run_times[s_ind_h, i_nb, 0, 0, 0] = t_lp_c_1
            run_times[s_ind_h, i_nb, 1, 0, 0] = t_lp_c_2

        BW_v1 = np.load(str_ident + '_val_bw_bc_1.npy')
        BW_v2 = np.load(str_ident + '_val_bw_bc_2.npy')
        t_bw_bc_1 = np.load(str_ident + '_time_bw_bc_1.npy')
        t_bw_bc_2 = np.load(str_ident + '_time_bw_bc_2.npy')
        values[s_ind_h, i_nb, 0, 1, 1] = BW_v1
        values[s_ind_h, i_nb, 1, 1, 1] = BW_v2
        run_times[s_ind_h, i_nb, 0, 1, 1] = t_bw_bc_1
        run_times[s_ind_h, i_nb, 1, 1, 1] = t_bw_bc_2

        BWs_v1 = np.load(str_ident + '_val_bws_bc_1.npy')
        BWs_v2 = np.load(str_ident + '_val_bws_bc_2.npy')
        t_bws_bc_1 = np.load(str_ident + '_time_bws_bc_1.npy')
        t_bws_bc_2 = np.load(str_ident + '_time_bws_bc_2.npy')
        values[s_ind_h, i_nb, 0, 1, 2] = BWs_v1
        values[s_ind_h, i_nb, 1, 1, 2] = BWs_v2
        run_times[s_ind_h, i_nb, 0, 1, 2] = t_bws_bc_1
        run_times[s_ind_h, i_nb, 1, 1, 2] = t_bws_bc_2
        relative_errors[s_ind_h, i_nb, 0, 1, 2] = np.abs(BWs_v1-BW_v1)/(val_prod_1-BW_v1)
        relative_errors[s_ind_h, i_nb, 1, 1, 2] = np.abs(BWs_v2-BW_v2)/(val_prod_2-BW_v2)

        # Sinkhorn
        for e_ind, EPS in enumerate(EPS_LIST):
            str_ident_eps = str_ident + '_' + str(EPS)

            sink_c_v1 = np.load(str_ident_eps + '_val_sink_c_1.npy')
            sink_c_v2 = np.load(str_ident_eps + '_val_sink_c_2.npy')
            t_sink_c_1 = np.load(str_ident_eps + '_time_sink_c_1.npy')
            t_sink_c_2 = np.load(str_ident_eps + '_time_sink_c_2.npy')
            if e_ind == 0:
                values[s_ind_h, i_nb, 0, 0, 3] = sink_c_v1
                values[s_ind_h, i_nb, 1, 0, 3] = sink_c_v2
                run_times[s_ind_h, i_nb, 0, 0, 3] = t_sink_c_1
                run_times[s_ind_h, i_nb, 1, 0, 3] = t_sink_c_2
                relative_errors[s_ind_h, i_nb, 0, 0, 3] = np.abs(sink_c_v1 - values[s_ind_h, i_nb, 0, 0, 0]) / (val_prod_1 - values[s_ind_h, i_nb, 0, 0, 0])
                relative_errors[s_ind_h, i_nb, 1, 0, 3] = np.abs(sink_c_v2 - values[s_ind_h, i_nb, 1, 0, 0]) / (val_prod_2 - values[s_ind_h, i_nb, 1, 0, 0])
            else:
                values[s_ind_h, i_nb, 0, 0, 4] = sink_c_v1
                values[s_ind_h, i_nb, 1, 0, 4] = sink_c_v2
                run_times[s_ind_h, i_nb, 0, 0, 4] = t_sink_c_1
                run_times[s_ind_h, i_nb, 1, 0, 4] = t_sink_c_2
                relative_errors[s_ind_h, i_nb, 0, 0, 4] = np.abs(sink_c_v1 - values[s_ind_h, i_nb, 0, 0, 0]) / (val_prod_1 - values[s_ind_h, i_nb, 0, 0, 0])
                relative_errors[s_ind_h, i_nb, 1, 0, 4] = np.abs(sink_c_v2 - values[s_ind_h, i_nb, 1, 0, 0]) / (val_prod_2 - values[s_ind_h, i_nb, 1, 0, 0])


            sink_bc_v1 = np.load(str_ident_eps + '_val_sink_bc_1.npy')
            sink_bc_v2 = np.load(str_ident_eps + '_val_sink_bc_2.npy')
            t_sink_bc_1 = np.load(str_ident_eps + '_time_sink_bc_1.npy')
            t_sink_bc_2 = np.load(str_ident_eps + '_time_sink_bc_2.npy')
            if e_ind == 0:
                values[s_ind_h, i_nb, 0, 1, 3] = sink_bc_v1
                values[s_ind_h, i_nb, 1, 1, 3] = sink_bc_v2
                run_times[s_ind_h, i_nb, 0, 1, 3] = t_sink_bc_1
                run_times[s_ind_h, i_nb, 1, 1, 3] = t_sink_bc_2
                relative_errors[s_ind_h, i_nb, 0, 1, 3] = np.abs(sink_bc_v1 - values[s_ind_h, i_nb, 0, 1, 1]) / (val_prod_1 - values[s_ind_h, i_nb, 0, 1, 1])
                relative_errors[s_ind_h, i_nb, 1, 1, 3] = np.abs(sink_bc_v2 - values[s_ind_h, i_nb, 1, 1, 1]) / (val_prod_2 - values[s_ind_h, i_nb, 1, 1, 1])
            else:
                values[s_ind_h, i_nb, 0, 1, 4] = sink_bc_v1
                values[s_ind_h, i_nb, 1, 1, 4] = sink_bc_v2
                run_times[s_ind_h, i_nb, 0, 1, 4] = t_sink_bc_1
                run_times[s_ind_h, i_nb, 1, 1, 4] = t_sink_bc_2
                relative_errors[s_ind_h, i_nb, 0, 1, 4] = np.abs(sink_bc_v1 - values[s_ind_h, i_nb, 0, 1, 1]) / (val_prod_1 - values[s_ind_h, i_nb, 0, 1, 1])
                relative_errors[s_ind_h, i_nb, 1, 1, 4] = np.abs(sink_bc_v2 - values[s_ind_h, i_nb, 1, 1, 1]) / (val_prod_2 - values[s_ind_h, i_nb, 1, 1, 1])

mrt = np.mean(run_times, axis=0)
mre = np.mean(relative_errors, axis=0)

for i_nb in range(len(N_BRANCH_LIST)):
    print('_________________________')
    print(i_nb)
    print(mrt[i_nb, 0, :, :])  # first cost
    print(mrt[i_nb, 1, :, :])  # second cost
    print(mre[i_nb, 0, :, :]*100)  # relative error in percent first cost
    print(mre[i_nb, 1, :, :]*100)  # relative error in percent second cost
