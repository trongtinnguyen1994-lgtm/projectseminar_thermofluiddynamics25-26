# import third party libraries
import jax
import jax.numpy as jnp
from pathlib import Path

# import own packages
from pstfd_project.data_handlers import load_KRNLV_data, prep_KRNLV_data_opt2, prep_valuation, KRNLV_train_test_split
from pstfd_project.jax_model_implementation import MyMLP
from pstfd_project.plotting import KRNLV_data_plot
from pstfd_project.general_utilities import norm2_KRNLV_DATA
from pstfd_project.model_functions import train
from pstfd_project.plotting import loss_plot
from pstfd_project.save_hyperparams import save_KRNLV_hyperparams, save_KRNLV_model


def main_FDF_opt2_train():

    # BEST MODEL: 1

    # ------- CUSTOM PARAMS -------
    # =============================

    # saving training data
    save_data = True

    # hyperparmeters
    WINDOW_SIZE = 50
    NO_BATCHES = 40
    learning_rate = 6e-4
    min_delta = 1e-3
    l_error_weight = 40
    weights_decay = 3e-4
    EPOCHS = 8000
    PATIENCE = 500
    no_neurons_h1 = 28
    no_neurons_h2 = 8
    UNDERSAMPLING = 150

    # Plot Kornilov data set?
    KRNLV_plot_bool = True
    # select data set (A: 10, 50, 100; T_max: 0.2, 0.4)
    KRNLV_data_set = [100, 0.4]

    # =========================
    # -------END CUSTOM -------

    #-----------prepare Kornilov data set-------------
    # get project directory
    DIR = Path(__file__).resolve().parent.parent

    # load Kornilov mean flow data as dictionary
    DIR_KRNLV = DIR / 'Data/Kornilov_Mean_flow_Ramp'
    KRNLV_DATA = load_KRNLV_data(path=DIR_KRNLV, u_rate=UNDERSAMPLING)

    # plotting vel / hrr over time
    if KRNLV_plot_bool:
        KRNLV_data_plot(KRNLV_DATA, KRNLV_data_set)    
    
    # choosing which A for [linear, nonlinear] data
    train_A = [10, 100]
    test_A = [10, 100]

    print('----------normalizing----------')
    # create train / test data (train: T_max=0.2, test: T_max=0.4)
    KRNLV_train, KRNLV_test = KRNLV_train_test_split(KRNLV_DATA, train_A, test_A)

    # normalize data with nonlinear train in data
    l_KRNLV_norm, nl_KRNLV_norm, norm_fac = norm2_KRNLV_DATA(KRNLV_train, KRNLV_test)

    print('----------prep data------------')
    # preparing linear data (window, batch)
    l_krn_data = prep_KRNLV_data_opt2(
        l_KRNLV_norm,
        WINDOW_SIZE,
        NO_BATCHES
        )
    
    # preparing nonlinear data (window, batch)
    nl_krn_data = prep_KRNLV_data_opt2(
        nl_KRNLV_norm,
        WINDOW_SIZE,
        NO_BATCHES
        )

    # structure of NN
    shape = [WINDOW_SIZE, no_neurons_h1, no_neurons_h2, 1]

    # set seed
    key = jax.random.PRNGKey(0)

    # creating model
    model = MyMLP(key, shape, learning_rate)

    # calc + plotting loss
    loss_history = train(model, 
                            nl_krn_data['train_in'], 
                            nl_krn_data['train_out'], 
                            l_krn_data['train_in'], 
                            l_krn_data['train_out'], 
                            EPOCHS, 
                            PATIENCE, 
                            min_delta,
                            l_error_weight,
                            weights_decay
                           )
    loss_plot(loss_history)

    # save params for trained model
    params = model.params
    
    print('----------prep results---------')
    # setting data up for valuation of training + test data (nonlinear / linear)
    nl_train_fit, nl_test_fit = prep_valuation(
        nl_krn_data['train_in'], 
        nl_krn_data['train_out'], 
        nl_krn_data['test_in'], 
        nl_krn_data['test_out'], 
        nl_krn_data['train_time'],
        nl_krn_data['test_time'],
        model,
        params,
        plot=False,
        is_nonlinear=True
        )
    
    l_train_fit, l_test_fit = prep_valuation(
        l_krn_data['train_in'], 
        l_krn_data['train_out'], 
        l_krn_data['test_in'], 
        l_krn_data['test_out'], 
        l_krn_data['train_time'],
        l_krn_data['test_time'],
        model,
        params,
        plot=False,
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

    # wether user wants to save or not
    if save_data:
        # preparing saving
        print('--------------saving model------------------')
        # creating directory + file
        model_dir = DIR / 'Documentation/KRNLV_data/opt2/saved_KRNLV_models_opt2'

        # find all already existing files
        existing_file = list(model_dir.glob('KRNLV_model_opt2_packages*.pk1')) #.glob() searches all files with matching patterns

        # set index for next file
        idx = len(existing_file) + 1


        # saving hyperparameters by adding to 'saved_hyperparameters.txt'
        file_path = Path(DIR/'Documentation/KRNLV_data/opt2/saved_KRNLV_opt2_hyperparameters.txt')
        hyperparams = save_KRNLV_hyperparams(
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
            nl_krn_data,
            nl_train_fit,
            nl_test_fit,
            l_train_fit,
            l_test_fit,
            idx
            )
        # saving model in pickle file
        # save path
        save_path = model_dir / f'KRNLV_model_opt2_packages{idx}.pk1'
        save_KRNLV_model(
            save_path,
            key,
            params,
            shape,
            hyperparams
        )

if __name__ == '__main__':
    main_FDF_opt2_train()