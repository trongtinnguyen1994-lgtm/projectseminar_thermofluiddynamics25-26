# importing third party packages
import jax
import jax.numpy as jnp
from pathlib import Path
import pickle

# importing own packages
from pstfd_project.jax_model_implementation import MyMLP
from pstfd_project.data_handlers import load_data, prep_data, prep_valuation


def main_MLP_test():

    # ------- CUSTOM PARAMS -------
    # =============================

    # select file number to test
    idx = 1

    # ==========================
    # ------- CUSTOM END -------

    # set seed
    key = jax.random.PRNGKey(0)

    # get directory path
    DIR = Path(__file__).resolve().parent.parent

    # loading nonlinear data
    NL_DATA = load_data(DIR/'Data/NonlinearFlameData.h5', u_rate=4)

    # loading linear data
    L_DATA = load_data(DIR/'Data/LinearFlameData.h5', u_rate=20)


    # loading pickle file from training
    # path directory to file
    path = Path(DIR / f'Documentation/NoRamp_data/saved_models/model_packages{idx}.pk1')

    # raise Error if file not found
    if not path.exists():
        raise FileNotFoundError(f'Pickel file for idx: {idx} not found at:\n {path}')

    # loading file
    with open(path, 'rb') as f:
        package = pickle.load(f)

    # loading data
    params = package['params']
    key = package['key']
    shape = package['shape']


    # hyperparmeters
    hyp = package['hyperparams']
    WINDOW_SIZE = hyp['window_size']
    NO_BATCHES = hyp['no_batches']
    learning_rate = hyp['learning_rate']
    no_neurons_h1 = hyp['no_neurons hidden layer 1']
    no_neurons_h2 = hyp['no_neurons hidden layer 2']
    brownian_noise_lvl = hyp['brownian_noise_lvl']
    brownian_noise_factor = hyp['brownian_noise_factor']
    white_noise_lvl = hyp['white_noise_lvl']
    white_noise_factor = hyp['white_noise_factor']
    
    print('----------prep data---------')
    # preparing nonlinear data (window, batch, noise)
    nl_data = prep_data(
        NL_DATA,
        WINDOW_SIZE,
        NO_BATCHES,
        brownian_noise_lvl, 
        brownian_noise_factor, 
        white_noise_lvl, 
        white_noise_factor
        )
    # prepare linear data
    l_data = prep_data(
        L_DATA,
        WINDOW_SIZE,
        NO_BATCHES,
        brownian_noise_lvl, 
        brownian_noise_factor, 
        white_noise_lvl, 
        white_noise_factor
        )

    # structure of NN
    shape = [WINDOW_SIZE, no_neurons_h1, no_neurons_h2, 1]

    # creating model
    model = MyMLP(key, shape, learning_rate)

    print('----------prep results---------')
    # setting data up for valuation of training + test data (nonlinear / linear)
    nl_train_fit, nl_test_fit = prep_valuation(
        nl_data['train_in'], 
        nl_data['train_out'], 
        nl_data['test_in'], 
        nl_data['test_out'], 
        nl_data['train_time'],
        nl_data['test_time'],
        model,
        params,
        plot=True,
        is_nonlinear=True
        )
    
    l_train_fit, l_test_fit = prep_valuation(
        l_data['train_in'], 
        l_data['train_out'], 
        l_data['test_in'], 
        l_data['test_out'], 
        l_data['train_time'],
        l_data['test_time'],
        model,
        params,
        plot=True,
        is_nonlinear=False
        )

    # train + test fit value (nonlinear / linear)
    print('\n################################################')
    print(f"Fit value for linear train data: {(float(jnp.round(l_train_fit, 2))):.2f}%")
    print(f"Fit value for linear test data:  {(float(jnp.round(l_test_fit, 2))):.2f}%")
    print('################################################')
    print(f"Fit value for nonlinear train data: {(float(jnp.round(nl_train_fit, 2))):.2f}%")
    print(f"Fit value for nonlinear test data:  {(float(jnp.round(nl_test_fit, 2))):.2f}%")
    print('################################################\n')


if __name__ == '__main__':
    main_MLP_test()