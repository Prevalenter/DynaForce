
import sys
sys.path.append('../')

from utils.identification.gene_robot import get_robot
from utils.identification.support_funcs.regr_data_proc import gen_regr_matrices


if __name__ == '__main__':
    for i in range(3):
        print('-'*30)
        rbt = get_robot(f'../data/model/gx11pm_finger{i+1}.pkl')
        for i, b in enumerate(rbt.dyn.baseparms):
            print(i, b)
        # row = ['%.15s ...'%b if len(str(b)) > 7 else str(b)]
        # row += [form%beta_init[i, 0]]
        # # row += [form%beta_prime[i,0]]
        # row += [form%beta_star[i, 0]]
        # print(i, ' & '.join(row), '\\\\')
        # table.append(row)