import numpy as np
import matplotlib.pyplot as plt

# this file is used to produce Figure 1 in the paper Eckstein&Pammer "Computational methods ..."

plt.rc('text', usetex=True)
plt.rcParams.update({
    'font.size': 18,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})


ss_each = 10
n_list = [20, 40, 75, 100, 200, 500, 1000, 1500, 2000, 3000, 5000, 7500, 10000]
space_list = list(range(1, len(n_list)+1))
space_list_ind = [1, 3, 5, 7, 9, 11, 13]
space_list_sparse = [20, 75, 200, 1000, 2000, 5000, 10000]
indexing = list(range(len(n_list)))

values_bc_emp = np.load('../data/values_bc_emp.npy')
values_bc_ademp = np.load('../data/values_bc_ademp.npy')
values_ot_emp = np.load('../data/values_ot_emp.npy')
values_ot_ademp = np.load('../data/values_ot_ademp.npy')
times_bc_emp = np.load('../data/times_bc_emp.npy')
times_bc_ademp = np.load('../data/times_bc_ademp.npy')
times_ot_emp = np.load('../data/times_ot_emp.npy')
times_ot_ademp = np.load('../data/times_ot_ademp.npy')

values_bc_emp_av = np.mean(values_bc_emp, axis=1)
values_bc_emp_std = np.std(values_bc_emp, axis=1)
values_ot_emp_av = np.mean(values_ot_emp, axis=1)
values_ot_emp_std = np.std(values_ot_emp, axis=1)
values_bc_ademp_av = np.mean(values_bc_ademp, axis=1)
values_bc_ademp_std = np.std(values_bc_ademp, axis=1)
values_ot_ademp_av = np.mean(values_ot_ademp, axis=1)
values_ot_ademp_std = np.std(values_ot_ademp, axis=1)

times_bc_emp_av = np.mean(times_bc_emp, axis=1)
times_bc_emp_std = np.std(times_bc_emp, axis=1)
times_ot_emp_av = np.mean(times_ot_emp, axis=1)
times_ot_emp_std = np.std(times_ot_emp, axis=1)
times_bc_ademp_av = np.mean(times_bc_ademp, axis=1)
times_bc_ademp_std = np.std(times_bc_ademp, axis=1)
times_ot_ademp_av = np.mean(times_ot_ademp, axis=1)
times_ot_ademp_std = np.std(times_ot_ademp, axis=1)

plt.plot(space_list[:-4], values_bc_emp_av[indexing][:-4], '-', label=r'$W_{1, bc}(\mu^n, \nu)$')
plt.fill_between(space_list[:-4], values_bc_emp_av[indexing][:-4]-values_bc_emp_std[indexing][:-4], values_bc_emp_av[indexing][:-4]+values_bc_emp_std[indexing][:-4], alpha=0.5)

plt.plot(space_list, values_bc_ademp_av[indexing], '--', label=r'$W_{1, bc}(\hat{\mu}^n, \nu)$')
plt.fill_between(space_list, values_bc_ademp_av[indexing]-values_bc_ademp_std[indexing], values_bc_ademp_av[indexing]+values_bc_ademp_std[indexing], alpha=0.5)

plt.xticks(space_list_ind, space_list_sparse)
plt.xlabel(r'$n$')
plt.legend()
plt.ylim([0.65, 1.65])
plt.gcf().subplots_adjust(bottom=0.13)
plt.savefig('../data/value_conv_41.png', format='png', dpi=500)
plt.show()


plt.plot(space_list, values_ot_emp_av[indexing], '-', label=r'$W_{1}(\mu^n, \nu)$')
plt.fill_between(space_list, values_ot_emp_av[indexing]-values_ot_emp_std[indexing], values_ot_emp_av[indexing]+values_ot_emp_std[indexing], alpha=0.5)

plt.plot(space_list, values_ot_ademp_av[indexing], '--', label=r'$W_{1}(\hat{\mu}^n, \nu)$')
plt.fill_between(space_list, values_ot_ademp_av[indexing]-values_ot_ademp_std[indexing], values_ot_ademp_av[indexing]+values_ot_ademp_std[indexing], alpha=0.5)

plt.xticks(space_list_ind, space_list_sparse)
plt.legend()
plt.xlabel(r'$n$')
plt.ylim([0.65, 1.65])
plt.gcf().subplots_adjust(bottom=0.13)
plt.savefig('../data/value_conv_42.png', format='png', dpi=500)
plt.show()

print(times_bc_ademp_av)
print(times_ot_emp_av)
plt.plot(space_list, times_bc_ademp_av[indexing], '--', label=r'Runtime for $W_{1, bc}(\hat{\mu}^n, \nu)$')

plt.plot(space_list, times_ot_emp_av[indexing], '-', label=r'Runtime for $W_{1}(\mu^n, \nu)$')

plt.xticks(space_list_ind, space_list_sparse)
plt.legend()
plt.ylabel('seconds')
plt.xlabel(r'$n$')
plt.gcf().subplots_adjust(bottom=0.13)
plt.savefig('../data/times_3.png', format='png', dpi=500)
plt.show()
