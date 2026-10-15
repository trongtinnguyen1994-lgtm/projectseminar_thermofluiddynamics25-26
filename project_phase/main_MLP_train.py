# importing third party packages
import jax
import jax.numpy as jnp
from pathlib import Path

# importing own packages
from pstfd_project.jax_model_implementation import MyMLP
from pstfd_project.model_functions import train
from pstfd_project.plotting import loss_plot
from pstfd_project.data_handlers import load_data, prep_data, prep_valuation
from pstfd_project.save_hyperparams import save_hyperparams, save_model


def main_MLP_train():

    # BEST MODEL: 1

    # ------- CUSTOM PARAMS -------
    # =============================

    # saving training model / hyperparams
    save_data = True

    # hyperparmeters
    WINDOW_SIZE = 50
    NO_BATCHES = 40
    learning_rate = 5e-5
    min_delta = 1e-3
    l_error_weight = 50
    weights_decay = 1e-3
    EPOCHS = 5000
    PATIENCE = 1000
    no_neurons_h1 = 25
    no_neurons_h2 = 5
    brownian_noise_lvl = 0.1
    brownian_noise_factor = 0.02
    white_noise_lvl = 0.1
    white_noise_factor = 0.02

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

    # calc + plotting loss (nonlinear + linear)
    loss_history = train(model, 
                           nl_data['train_in'], 
                           nl_data['train_out'], 
                           l_data['train_in'], 
                           l_data['train_out'], 
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
        nl_data['train_in'], 
        nl_data['train_out'], 
        nl_data['test_in'], 
        nl_data['test_out'], 
        nl_data['train_time'],
        nl_data['test_time'],
        model,
        params,
        plot=False,
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
        model_dir = DIR / 'Documentation/NoRamp_data/saved_models'

        # find all already existing files
        existing_file = list(model_dir.glob('model_packages*.pk1'))

        # set index for next file
        idx = len(existing_file) + 1

        # saving respective hyperparameters
        file_path = Path(DIR/'Documentation/NoRamp_data/saved_hyperparameters.txt')
        hyperparams = save_hyperparams(
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
            nl_data,
            nl_train_fit,
            nl_test_fit,
            l_train_fit,
            l_test_fit,
            idx
            )
        # save path
        save_path = model_dir / f'model_packages{idx}.pk1'

        # saving model in pickle file
        save_model(
            save_path,
            key,
            params,
            shape,
            hyperparams
            )
    
if __name__ == '__main__':
    main_MLP_train()