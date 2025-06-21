
import matplotlib.pyplot as plt
import numpy as np

def correlation_coefficient(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0, 1]

def mse(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

from sklearn.metrics import mean_absolute_percentage_error as mape


if __name__ == '__main__':
    
    data_dir = 'data/thumb_vertical_long'
    
    
    pred_force_no_ident = np.load(f'{data_dir}/force_pred_no_ident.npy')[1010:1310]
    
    force_pred_raw_list = np.load(f'{data_dir}/force_pred_raw_list.npy')
    # load the data
    force_pred_raw_list_momentum = np.load(f'{data_dir}/force_pred_raw_list_momentum_{2.5}.npy')[:300]
    force = np.load(f'{data_dir}/force_list.npy')[1010:1310]
    force = np.abs(force)[:, None]
    
    # force_pred_with_c_raw_list_norm = np.linalg.norm(force_pred_with_c_raw_list, axis=1)*10
    force_pred_norm = np.linalg.norm(force_pred_raw_list, axis=1)[1000:1300]*10
    force_pred_momentum_norm = np.linalg.norm(force_pred_raw_list_momentum, axis=1)*10
    pred_force_no_ident_norm = np.linalg.norm(pred_force_no_ident, axis=1)[:, None]*10
    
    
    print(force_pred_norm.shape, force_pred_momentum_norm.shape, force.shape, pred_force_no_ident_norm.shape)
    
    
    force += 1e-2
    print('-'*40)
    print('mse')
    print(mse(force.flatten(), pred_force_no_ident_norm.flatten()))
    print(mse(force.flatten(), force_pred_norm.flatten()))
    print(mse(force.flatten(), force_pred_momentum_norm.flatten()))
    # print(mse(force, pred_force_no_ident_norm))

    print('-'*40)
    print('mape')
    print(mape(force.flatten(), pred_force_no_ident_norm.flatten()))
    print(mape(force.flatten(), force_pred_norm.flatten()))
    print(mape(force.flatten(), force_pred_momentum_norm.flatten()))

    print('-'*40)
    # print('correlation_coefficient')
    # # 添加调试信息
    # print("force stats:", np.min(force), np.max(force), np.std(force))
    # print("pred_force_no_ident_norm stats:", np.min(pred_force_no_ident_norm), np.max(pred_force_no_ident_norm), np.std(pred_force_no_ident_norm))
    # 尝试展平数组
    print(correlation_coefficient(force.flatten(), pred_force_no_ident_norm.flatten()))
    print(correlation_coefficient(force.flatten(), force_pred_norm.flatten()))
    print(correlation_coefficient(force.flatten(), force_pred_momentum_norm.flatten()))