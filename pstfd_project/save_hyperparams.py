from datetime import datetime
import pickle


def save_hyperparams(
        file_path,
        WINDOW_SIZE,
        NO_BATCHES,
        learning_rate,
        min_delta,
        l_error_weight,
        weights_decay,       
        EPOCHS,
        PATIENCE,
        no_neurons_h1,
        no_neurons_h2,
        brownian_noise_lvl,
        brownian_noise_factor,
        white_noise_lvl,
        white_noise_factor,
        data,
        nl_train_fit,
        nl_test_fit,
        l_train_fit,
        l_test_fit,
        idx
          ):
        """
        saves all hyperparameters and fit values from training into corresponding txt-file 
        for data on constant mean velocity
        """

        # setting up current hyperparams
        new_hyperparams= {
            'window_size': WINDOW_SIZE,
            'no_batches': NO_BATCHES,
            'learning_rate': learning_rate,
            'min_delta': min_delta,
            'l_error_weight': l_error_weight,
            'weights_decay': weights_decay,
            'epochs': EPOCHS,
            'patience': PATIENCE,
            'no_neurons hidden layer 1': no_neurons_h1,
            'no_neurons hidden layer 2': no_neurons_h2,
            'brownian_noise_lvl': brownian_noise_lvl,
            'brownian_noise_factor': brownian_noise_factor,
            'white_noise_lvl': white_noise_lvl,
            'white_noise_factor': white_noise_factor,
            'ORDER': int(data['ORDER']),
            'LINEAR TRAIN FIT': f"{float(l_train_fit):.2f}" + '%',
            'LINEAR TEST FIT': f"{float(l_test_fit):.2f}" + '%',
            'NONLINEAR TRAIN FIT': f"{float(nl_train_fit):.2f}" + '%',
            'NONLINEAR TEST FIT': f"{float(nl_test_fit):.2f}" + '%'
        }

        with open(file_path, 'a') as f:
                f.write('----------HYPERPARAMETERS----------\n')
                f.write(f'----MATCHING PICKLE FILE NO: {idx}----\n')
                f.write(f"date / time: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}")
                f.write('\n')
                for key, value in new_hyperparams.items():
                    f.write(f"{key}: {value}\n")
                f.write('\n')
        return new_hyperparams


def save_model(save_path, key, params, shape, hyperparams):
    """
    saves model and model related information into pickle file
    for data on constant mean velocity
    """
       
    save_dict = {
        'saved at': datetime.now().strftime('%d-%m-%Y %H:%M:%S'),
        'params': params,
        'key': key,
        'shape': shape,
        'hyperparams': hyperparams,
       }
    with open(save_path, 'wb') as f:
          pickle.dump(save_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved training package to: {save_path}")


def save_KRNLV_hyperparams(
        file_path,
        WINDOW_SIZE,
        NO_BATCHES,
        learning_rate,
        min_delta,
        l_error_weight,
        weights_decay,       
        EPOCHS,
        PATIENCE,
        no_neurons_h1,
        no_neurons_h2,
        UNDERSAMPLING,
        data,
        nl_train_fit,
        nl_test_fit,
        l_train_fit,
        l_test_fit,
        idx
          ):
        """
        saves all hyperparameters and fit values from training into corresponding txt-file 
        for Kornilov data
        """

        # setting up current hyperparams
        new_hyperparams= {
            'window_size': WINDOW_SIZE,
            'no_batches': NO_BATCHES,
            'learning_rate': learning_rate,
            'min_delta': min_delta,
            'l_error_weight': l_error_weight,
            'weights_decay': weights_decay,
            'epochs': EPOCHS,
            'patience': PATIENCE,
            'no_neurons hidden layer 1': no_neurons_h1,
            'no_neurons hidden layer 2': no_neurons_h2,
            'undersampling': UNDERSAMPLING,
            'ORDER': int(data['ORDER']),
            'LINEAR TRAIN FIT': f"{float(l_train_fit):.2f}" + '%',
            'LINEAR TEST FIT': f"{float(l_test_fit):.2f}" + '%',
            'NONLINEAR TRAIN FIT': f"{float(nl_train_fit):.2f}" + '%',
            'NONLINEAR TEST FIT': f"{float(nl_test_fit):.2f}" + '%'
            
        }

        with open(file_path, 'a') as f:
                f.write('----------HYPERPARAMETERS----------\n')
                f.write(f'----MATCHING PICKLE FILE NO: {idx}----\n')
                f.write(f"date / time: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}")
                f.write('\n')
                for key, value in new_hyperparams.items():
                    f.write(f"{key}: {value}\n")
                f.write('\n')
        return new_hyperparams


def save_KRNLV_model(save_path, key, params, shape, hyperparams):
    """
    saves model and model related information into pickle file
    for Kornilov data
    """
       
    save_dict = {
        'saved at': datetime.now().strftime('%d-%m-%Y %H:%M:%S'),
        'params': params,
        'key': key,
        'shape': shape,
        'hyperparams': hyperparams,
       }
    with open(save_path, 'wb') as f:
          pickle.dump(save_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved training package to: {save_path}")