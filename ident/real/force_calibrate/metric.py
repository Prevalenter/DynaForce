
import matplotlib.pyplot as plt
import numpy as np

# import correlation coefficient
def correlation_coefficient(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0, 1]

# import the relative error
def relative_error(y_true, y_pred):
    return np.abs((y_true - y_pred) / (y_true+6e-1))

def mse(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()


if __name__ == '__main__':
    # load the data

    data_dir = 'data/thumb_vertical_long' # 'data/thumb_horizontal'
    pos = np.load(f'{data_dir}/pos_list.npy')[10:]
    torque = np.load(f'{data_dir}/torque_list.npy')[10:]
    time_list = np.load(f'{data_dir}/t_list.npy')[10:]
    force = np.load(f'{data_dir}/force_list.npy')[10:]
    pred_force_list = np.load(f'{data_dir}/force_pred_list.npy')
    pred_force_no_smooth = np.load(f'{data_dir}/force_pred_raw_list.npy')
    pred_force_no_ident = np.load(f'{data_dir}/force_pred_no_ident.npy')[10:]
    
    force = np.abs(force)

    
    print(pos.shape, torque.shape, time_list.shape, force.shape, pred_force_list.shape, 
          pred_force_no_smooth.shape, pred_force_no_ident.shape)

    print(force.max())
    
    print(pred_force_list.max(), pred_force_no_smooth.max(), pred_force_no_ident.max())
    
    pred_force_list_norm = np.linalg.norm(pred_force_list, axis=(1, 2))
    pred_force_no_smooth_norm = np.linalg.norm(pred_force_no_smooth, axis=(1, 2))
    pred_force_no_ident_norm = np.linalg.norm(pred_force_no_ident, axis=(1))
        
    # 
    
    print(pred_force_list_norm.shape, pred_force_no_smooth_norm.shape, pred_force_no_ident_norm.shape)
    print(pred_force_list_norm.max(), pred_force_no_smooth_norm.max(), pred_force_no_ident_norm.max())
    
    scale = np.abs(force[10:]).mean()/np.abs(pred_force_list_norm).mean()
    
    print(scale)
    
    pred_force_list_norm *= scale
    pred_force_no_smooth_norm *= scale
    pred_force_no_ident_norm *= scale

    print(pred_force_list_norm.max(), pred_force_no_smooth_norm.max(), pred_force_no_ident_norm.max())
    
    print('-'*40)
    
    print(mse(force, pred_force_list_norm))
    print(mse(force, pred_force_no_smooth_norm))
    print(mse(force, pred_force_no_ident_norm))
    
    print('-'*40)
    # print the relative error
    print(relative_error(force, pred_force_list_norm).mean())
    print(relative_error(force, pred_force_no_smooth_norm).mean())
    print(relative_error(force, pred_force_no_ident_norm).mean())

    print('-'*40)
    # print the correlation coefficient
    print(correlation_coefficient(force, pred_force_list_norm))
    print(correlation_coefficient(force, pred_force_no_smooth_norm))
    print(correlation_coefficient(force, pred_force_no_ident_norm))

