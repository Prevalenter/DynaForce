import numpy as np
import matplotlib.pyplot as plt

def correlation_coefficient(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0, 1]

# import the relative error
def relative_error(y_true, y_pred):
    return np.abs((y_true - y_pred) / (y_true+6e-1))

def mse(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

from sklearn.metrics import mean_absolute_percentage_error as mape



if __name__=="__main__":
    data_dir = 'data/thumb_vertical_long'

    k_list = [0.1, 0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 8.0, 10.0]
    direct_mse_list = []
    momentum_mse_list = []
    momentum_mape_list = []
    momentum_corr_list = []
    direct_mape_list = []
    direct_corr_list = []

    

    for k in k_list:

        # force_pred_with_c_raw_list = np.load(f'{data_dir}/force_pred_raw_list_momentum_with_c.npy')
        
        # force_pred_list = np.load(f'{data_dir}/force_pred_list.npy')
        force_pred_raw_list = np.load(f'{data_dir}/force_pred_raw_list.npy')
        
        # force_pred_list_momentum = np.load(f'{data_dir}/force_pred_list_momentum.npy')
        force_pred_raw_list_momentum = np.load(f'{data_dir}/force_pred_raw_list_momentum_{k}.npy')[:300]
        
        force = np.load(f'{data_dir}/force_list.npy')[1010:1310]
        force = np.abs(force)[:, None]
        
        # force_pred_with_c_raw_list_norm = np.linalg.norm(force_pred_with_c_raw_list, axis=1)*10
        force_pred_norm = np.linalg.norm(force_pred_raw_list, axis=1)[1000:1300]*10
        force_pred_momentum_norm = np.linalg.norm(force_pred_raw_list_momentum, axis=1)*10

        print(force_pred_norm.shape, force_pred_momentum_norm.shape, force.shape)


        momentum_mse_list.append(mse(force[:], force_pred_momentum_norm))
        direct_mse_list.append(mse(force[:], force_pred_norm))
        print(momentum_mse_list[-1])
        
        # # plt.plot(force_pred_with_c_raw_list_norm, c='k', label='force_pred_with_c_raw_list_norm')
        # plt.plot(force_pred_norm, c='r', label='force_pred_norm')
        # plt.plot(force_pred_momentum_norm, c='b', label='force_pred_momentum_norm')
        # plt.plot(force[:], c='g', label='force')
        # plt.legend()
        # # print(force_pred_list.shape, force_pred_list_momentum.shape)
        
        # plt.show()
        
        # breakpoint()

        mape_value = mape(force+1e-2, force_pred_momentum_norm)
        corr_coef = correlation_coefficient(force.flatten(), force_pred_momentum_norm.flatten())
        momentum_mape_list.append(mape_value)
        momentum_corr_list.append(corr_coef)

        # Direct metrics
        direct_mape = mape(force+1e-2, force_pred_norm)
        direct_corr = correlation_coefficient(force.flatten(), force_pred_norm.flatten())
        direct_mape_list.append(direct_mape)
        direct_corr_list.append(direct_corr)

    # --- Subplots for MSE, MAPE, and Correlation Coefficient ---
    fig, axs = plt.subplots(1, 3, figsize=(10, 3), sharex=True)

    axs[0].plot(k_list, momentum_mse_list, '-o', label='momentum', markersize=8, alpha=0.7)
    axs[0].plot(k_list, direct_mse_list, '--', label='direct', markersize=8, alpha=0.7)
    axs[0].set_ylabel('MSE')
    axs[0].legend()
    axs[0].set_title('Mean Squared Error')

    axs[1].plot(k_list, momentum_mape_list, '-o', label='momentum', markersize=8, alpha=0.7)
    axs[1].plot(k_list, direct_mape_list, '--', label='direct', markersize=8, alpha=0.7)
    axs[1].set_ylabel('MAPE')
    axs[1].legend()
    axs[1].set_title('Mean Absolute Percentage Error')

    axs[2].plot(k_list, momentum_corr_list, '-o', label='momentum', markersize=8, alpha=0.7)
    axs[2].plot(k_list, direct_corr_list, '--', label='direct', markersize=8, alpha=0.7)
    axs[2].set_ylabel('Correlation Coefficient')
    axs[2].set_xlabel('K')
    axs[2].legend()
    axs[2].set_title('Correlation Coefficient')

    # plt.tight_layout()
    plt.show()
