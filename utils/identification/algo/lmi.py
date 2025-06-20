import lmi_sdp
import numpy as np
import matplotlib.pyplot as plt
import sympy
from sympy import BlockMatrix, Matrix, eye, Identity
import time
import sys
# sys.path.append('..')
sys.path.append('../../..')
# utils.identification.algo
# from utils import process_data, gene_robot
from utils.identification.support_funcs.regr_data_proc import gen_regr_matrices
from utils.identification.support_funcs.utils import skew, mrepl
from lmi_sdp import LMI_PD, LMI

from utils.identification.algo.ols import LeastSquares

epsilon_safemargin = 1e-6
epsilon_sdptol = 1e-7

def sdpa_file(objf, lmis, variables):
	sdpadat = lmi_sdp.to_sdpa_sparse(objf, lmis, variables)
	with open('../data/sdpa_dat/sdpa_input.dat-s', 'w') as f:
		f.write(sdpadat)
	print("SDPA file saved at: %ssdpa_dat/sdpa_input.dat-s" % tmpfolder)

def cvxopt_dsdp5(objf, lmis, variables):
    import cvxopt.solvers
    c, Gs, hs = lmi_sdp.to_cvxopt(objf, lmis, variables)
    cvxopt.solvers.options['DSDP_GapTolerance'] = epsilon_sdptol
    tic = time.time()
    sdpout = cvxopt.solvers.sdp(c, Gs=Gs, hs=hs, solver='dsdp')
    toc = time.time()
    print(sdpout['status'], ('ATT!: \'optimal\' does not necessarlly means feasible'))
    print('Elapsed time: %.2f s'%(toc-tic))
    return np.matrix(sdpout['x'])

def FBPE(W, omega, R1, rho1, rbt, beta_init=None):
	dof = rbt.dof
	delta = rbt.dyn.dynparms
	n_delta = rbt.dyn.n_dynparms
	beta = rbt.dyn.baseparms.n()
	n_beta = rbt.dyn.n_base
	beta_symbs = sympy.Matrix([sympy.Symbol('beta' + str(i + 1), real=True) for i in range(n_beta)])
	delta_d = (rbt.dyn.Pd.T * delta)
	n_delta_d = len(delta_d)
	Pb = rbt.dyn.Pb
	varchange_dict = dict(zip(Pb.T * delta, beta_symbs - (beta - Pb.T * delta)))

	I = Identity
	S = skew

	D_inertia_blocks = []
	for i in range(dof):
		L = rbt.rbtdef.L[i]
		l = rbt.rbtdef.l[i]
		m = rbt.rbtdef.m[i]
		Di = BlockMatrix([[L,    S(l).T],
		                  [S(l), I(3)*m]])
		D_inertia_blocks.append(Di.as_explicit())

	# 摩擦力的约束
	D_other_blocks = []
	for i in range(dof):
		if rbt.rbtdef.driveinertiamodel == 'simplified':
			D_other_blocks.append(Matrix([rbt.rbtdef.Ia[i]]))
		if 'viscous' in rbt.rbtdef.frictionmodel:
			D_other_blocks.append(Matrix([rbt.rbtdef.fv[i]]))
		if 'Coulomb' in rbt.rbtdef.frictionmodel:
			D_other_blocks.append(Matrix([rbt.rbtdef.fc[i]]))

	D_blocks = D_inertia_blocks + D_other_blocks

	varchange_dict = dict(zip(Pb.T * delta, beta_symbs - (beta - Pb.T * delta)))
	DB_blocks = [mrepl(Di, varchange_dict) for Di in D_blocks]
	DB_LMIs = list(map(LMI_PD, DB_blocks))

	DB_LMIs_marg = list(map(lambda lm: LMI(lm, epsilon_safemargin * eye(lm.shape[0])), DB_blocks))
	# print('omega', omega.shape)
	
	if beta_init is None:
		beta_init = LeastSquares(W, omega)

	# bulid the constrain for BPFC
	u = sympy.Symbol('u')
	U_beta = BlockMatrix([[Matrix([u]), (beta_init - beta_symbs).T],
	                      [beta_init - beta_symbs, I(n_beta)]])
	U_beta = U_beta.as_explicit()
	lmis_ols_bpfc = [LMI(U_beta)] + DB_LMIs_marg

	variables_ols_bpfc = [u] + list(beta_symbs) + list(delta_d)

	objf_ols_bpfc = u
	sol_ols_bpfc = cvxopt_dsdp5(objf_ols_bpfc, lmis_ols_bpfc, variables_ols_bpfc)

	u_prime = sol_ols_bpfc[0,0]
	beta_prime = sol_ols_bpfc[1:1+n_beta]
	delta_d_prime = sol_ols_bpfc[1+n_beta:]

	# build constrian for FBPE-OLS
	u = sympy.Symbol('u')
	rho2_norm_sqr = np.linalg.norm(omega - W * beta_init) ** 2
	U_rho = BlockMatrix([[Matrix([u - rho2_norm_sqr]), (rho1 - R1 * beta_symbs).T],
	                     [rho1 - R1 * beta_symbs, I(n_beta)]])
	U_rho = U_rho.as_explicit()
	lmis_fbpe_ols = [LMI(U_rho)] + DB_LMIs_marg
	variables_fbpe_ols = [u] + list(beta_symbs) + list(delta_d)
	objf_fbpe_ols = u
	sol_fbpe_ols = cvxopt_dsdp5(objf_fbpe_ols, lmis_fbpe_ols, variables_fbpe_ols)
	u_star = sol_fbpe_ols[0, 0]
	beta_star = np.matrix(sol_fbpe_ols[1:1 + n_beta])
	delta_d_star = np.matrix(sol_fbpe_ols[1 + n_beta:])

	return beta_init, beta_prime, beta_star





if __name__ == '__main__':
	# rbt = gene_robot.get_robot('../data/estun/estun_model_mdh.pkl')
	# print(rbt.dyn.n_dynparms, rbt.dyn.n_base)
	# # print(rbt.dyn.baseparms.n())
	#
	# torque_weight = [2.27, 2.27, 1.27, 0.335,0.318,0.285]
	# # circle_interploted sixJointSinCos2  1JointSinCos3 1JointSinCos4 1JointSinCos5 sixJointSinCos2 ET_1
	# q, qd, qdd, tau = process_data.get_data('../data/1JointSinCos3.txt')
	# # tau[:, :3] = -tau[:, :3]
	#
	# # q, qd, qdd, tau = process_data.get_sim_data()

	rbt = gene_robot.get_robot('../data/estun/estun_model_mdh.pkl')
	q, qd, qdd, tau = np.load('../data/data_train/data_masked.npy')


	W, omega, Q1, R1, rho1 = gen_regr_matrices(rbt, q, qd, qdd, tau)
	omega_draw = omega.copy().reshape((-1, 6))

	# print(omega.shape)
	# breakpoint()
	beta_init, beta_prime, beta_star = FBPE(W, omega, R1, rho1, rbt)

	from utils.support_funcs.utils import ListTable
	form = '%.3g'
	table = ListTable()
	header = ['', 'B_ols', 'B^']
	table.append(header)
	for i, b in enumerate(rbt.dyn.baseparms):
		# print(i, b)
		row = ['%.15s ...'%b if len(str(b)) > 7 else str(b)]
		row += [form%beta_init[i, 0]]
		# row += [form%beta_prime[i,0]]
		row += [form%beta_star[i, 0]]
		print(i, ' & '.join(row), '\\\\')
		table.append(row)



