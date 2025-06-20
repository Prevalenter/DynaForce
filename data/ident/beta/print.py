import numpy as np

if __name__=="__main__":
    algo_dict = {
        'ransac': 0,
        'wls': 1,
        'irls': 2,
        'lmi': 3,
    }
    beta = []
    for algo in list(algo_dict.keys()):
        beta_algo = []
        for joint_idx in range(3):
            print(algo, joint_idx)
            
            file_name = f'{algo}_beta_finger_{joint_idx}.npy'

            data = np.load(file_name)
            
            beta_algo.append(data)
            
        beta_algo = np.concatenate(beta_algo)
        beta.append(beta_algo.copy())
        print(beta_algo)
        
    beta = np.array(beta)[:, :, 0].T
    
    beta_print = (beta*1000).round(2)
    # 不要用科学记数法
    np.set_printoptions(suppress=True)
    print(beta_print[:, -1])
            