# import third party libraries
import jax
import jax.numpy as jnp
from pathlib import Path
import pickle
import matplotlib.pyplot as plt

# import own packages
from pstfd_project.data_handlers import load_data, get_gain_phase_xFDF, load_KRNLV_data, prep_KRNLV_data_opt1, prep_valuation, KRNLV_train_test_split, prep_forcing_signal, load_data, load_xFDF_data
from pstfd_project.jax_model_implementation import MyMLP
from pstfd_project.model_functions import model_predict_forc_xFDF
from pstfd_project.general_utilities import norm1_KRNLV_DATA, fit_value_jax, get_gain_phase


def main_FDF_opt1_test():

    # ------- CUSTOM PARAMS -------
    # =============================

    # pick desired model to be loaded
    idx = 1

    # plot time series model fit vs real
    plot = True
    
    # print fit values
    fit_values = True

    # plot + calc no ramp data mean_vel = 0.7 for validation of linear performance of Kornilov data
    validation_07 = True

    # verify model with FDF analysis
    FDF_bool = True

    # FDF in one big figure (vs multiple small figures)
    FDF_fig = True

    # set mean velocity for FDF analysis
    vel_mean = [0.4, 0.7]

    # =========================
    # -------END CUSTOM -------

    # get project directory
    DIR = Path(__file__).resolve().parent.parent

    # load model and params from training
    # path directory to file
    path = Path(DIR / f'Documentation/KRNLV_data/opt1/saved_KRNLV_models_opt1/KRNLV_model_opt1_packages{idx}.pk1')

    if not path.exists():
        raise FileNotFoundError(f'Pickle file for idx: {idx} not found at:\n {path}')
    
    # loading file
    with open(path, 'rb') as f:
        package = pickle.load(f)

    # loading data
    params = package['params']
    key = package['key']
    shape = package['shape']

    # hyperparameters
    hyp = package['hyperparams']
    WINDOW_SIZE = hyp['window_size']
    NO_BATCHES = hyp['no_batches']
    learning_rate = hyp['learning_rate']
    no_neurons_h1 = hyp['no_neurons hidden layer 1']
    no_neurons_h2 = hyp['no_neurons hidden layer 2']
    ORDER = hyp['ORDER']
    UNDERSAMPLING = hyp['undersampling']

    # load Kornilov data
    DIR_KRNLV = DIR / 'Data/Kornilov_Mean_flow_Ramp'
    KRNLV_DATA = load_KRNLV_data(path=DIR_KRNLV, u_rate=UNDERSAMPLING)

    # choosing which A for [linear, nonlinear] data
    train_A = [10, 100]
    test_A = [10, 100]

    print('----------normalizing----------')
    # create train / test data (train: T_max=0.2, test: T_max=0.4)
    KRNLV_train, KRNLV_test = KRNLV_train_test_split(KRNLV_DATA, train_A, test_A)

    # normalize data
    l_KRNLV_norm, nl_KRNLV_norm = norm1_KRNLV_DATA(KRNLV_train, KRNLV_test)

    print('----------prep data------------')
    # linear data
    l_krn_data = prep_KRNLV_data_opt1(
        l_KRNLV_norm,
        WINDOW_SIZE,
        NO_BATCHES
        )
    
    # nonlinear data
    nl_krn_data = prep_KRNLV_data_opt1(
        nl_KRNLV_norm,
        WINDOW_SIZE,
        NO_BATCHES
        )
    
    # structure of NN
    shape = [(WINDOW_SIZE + 2), no_neurons_h1, no_neurons_h2, 1]

    # set seed
    key = jax.random.PRNGKey(0)

    # creating model
    model = MyMLP(key, shape, learning_rate)

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
        plot,
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
        plot,
        is_nonlinear=False
        )
    
    # --------validate model for linear u_mean=7, A=.005----------
    
    from pstfd_project.plotting import prediction_plot
    from pstfd_project.general_utilities import create_windows


    if validation_07:
        # load other reference data to validate FDF
        v7_A005_DATA = load_data(DIR/'Data/AddData/A005_U07_Kornilov.h5', u_rate=UNDERSAMPLING)

        # prep
        TEST_IN = v7_A005_DATA['input_data']
        TEST_OUT = v7_A005_DATA['output_data']
        TEST_TIME = jnp.arange(0, v7_A005_DATA['ts'] * len(v7_A005_DATA['input_data']), v7_A005_DATA['ts'])

        ORDER = nl_krn_data['ORDER']

        TEST_IN = create_windows(TEST_IN, WINDOW_SIZE)

        TEST_IN = TEST_IN[:-ORDER]

        if len(TEST_TIME) > len(TEST_IN):
            TEST_TIME = TEST_TIME[:len(TEST_IN)]

        TEST_OUT = TEST_OUT[ORDER:]

        mean_vel_07 = jnp.full((TEST_IN.shape[0],2), 0.7)

        # adding start / end of mean to every window of signal
        TEST_IN = jnp.concatenate([TEST_IN, mean_vel_07], axis=1)

        test_in_full = jnp.vstack(TEST_IN).astype(jnp.float32)
        test_out_full = jnp.vstack(TEST_OUT).astype(jnp.float32)

        y_pred_test = model(params, test_in_full)

        y_pred_test_flat = jnp.ravel(y_pred_test)
        test_out_flat = jnp.ravel(test_out_full)

        test_fit = fit_value_jax(test_out_flat, y_pred_test_flat)

        prediction_plot(TEST_TIME, test_out_flat, y_pred_test_flat, is_nonlinear=False, is_train=False)

        print(f"Fit value for 0.7 validation test data:  {(float(jnp.round(test_fit, 2))):.2f}%")

    
    
    if fit_values:
        # train + test fit value (nonlinear / linear)
        print('\n################################################')
        print(f"Fit value for linear train data: {(float(jnp.round(l_train_fit, 2))):.2f}%")
        print(f"Fit value for linear test data:  {(float(jnp.round(l_test_fit, 2))):.2f}%")
        print('################################################')
        print(f"Fit value for nonlinear train data: {(float(jnp.round(nl_train_fit, 2))):.2f}%")
        print(f"Fit value for nonlinear test data:  {(float(jnp.round(nl_test_fit, 2))):.2f}%")
        print('################################################')


    if FDF_bool:
        print('----------FDF analysis---------')
        # load xFDF (Kornilov reference data)
        DIR_xFDF = DIR / 'Data/Kornilov_reference_xFDF'
        xFDF_FULL_DATA = load_xFDF_data(path=DIR_xFDF)

        # throw out harmonic 2, 3 of data set
        xFDF_DATA = [d for d in xFDF_FULL_DATA if d.get('harmonic') != (2, 3)]

        '''
        extracted data:
        freq: 100, 170, 200, 300, 400, 500
        amp: 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.5, 0.75, 0.8, 1.0, 1.25, 1.5
        '''
        # set all freq / amps to extract of reference data
        frequency = [25, 50, 75, 100, 170, 200, 300]
        forcing = [0.1, 0.5, 1.5]
        
        # create gain / phase of reference data
        # ! structure of gain / phase array: jnp.array(row:freq, col:amps)
        gain, phase = get_gain_phase_xFDF(xFDF_DATA, frequency, forcing)

        # get length of vel_mean for plotting later
        len_velmean = len(vel_mean)

        # one big figure
        if FDF_fig:
            plt.figure(figsize=(14, 8))

        print('----------processing-----------')
        for k, vel in enumerate(vel_mean):
            # clear data from previous mean vel
            gain_result = []
            phase_result = []
            for i, frc in enumerate(forcing):
                for j, frq in enumerate(frequency):

                    # -------setting up time vector: cutting transient and adjust signal to full periods----------
                    dt = (l_krn_data['train_time'][2] - l_krn_data['train_time'][1]) #=0.0003

                    # number of periods of input signal    
                    num_periods = 8
                    
                    # assumed length of transient
                    length_transient = 0.015 # [s]

                    # setting length of single period
                    length_period = 1 / (frq * dt)

                    # setting length of whole period
                    length_period_steps = int(jnp.round(num_periods * length_period))

                    # setting steps of transient
                    length_transient_steps = int(jnp.round(length_transient / dt))

                    # make sure that length of transient is multiple of length of period
                    if int(jnp.round(length_period)) < length_transient_steps:
                        add_period = int(jnp.ceil(length_transient_steps / length_period))
                        length_transient_steps = int(jnp.round(add_period * length_period))
                    else:
                        length_transient_steps = int(jnp.round(length_period))
                        add_period = 1
                    
                    # computing time vector
                    t_vec = jnp.arange(-ORDER * dt, (length_period_steps + length_transient_steps) * dt, dt)


                    # -----model prediction-----
                    # compute forcing signal
                    forc_signal = prep_forcing_signal(frq, frc, t_vec)

                    # predict output on input forcing signal via model
                    q_pred = model_predict_forc_xFDF(model, params, forc_signal, WINDOW_SIZE, vel)
                    
                    # cut of transient of model prediction
                    q_pred_flat = jnp.squeeze(q_pred[length_transient_steps:])

                    # get gain / phase of corresponding frquency
                    gain_model, phase_model = get_gain_phase(q_pred_flat, dt, frq, frc)

                    # add to corresponding list
                    gain_result.append(gain_model)
                    phase_result.append(phase_model)


            # all shape(40, )
            gain_result_array = jnp.array(gain_result)
            phase_result_array = jnp.array(phase_result)

            len_frq1 = len(frequency)

            col_vec = ["#0000FF", "#32CD32", "#DC143C", "#FF8C00", "#9400D3"]

            # plot reference gain vs model gain
            plt.subplot(len_velmean, 2, 2*k+1)
            for i in range(len(forcing)):
                start = len_frq1*i
                end = len_frq1*(i+1)
                plt.plot(frequency, gain_result_array[start : end], label=f'Model A = {forcing[i]}', marker='o', linewidth=0.8, color=col_vec[i])
                plt.plot(frequency, gain[:, i], label=f'True CFD A = {forcing[i]}', ls='--', marker='x', linewidth=0.8, color=col_vec[i])
            plt.title(f'Comparison Gain: Model vs CFD data || mean vel {vel} m/s')
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Gain [-]')
            plt.legend()
            plt.grid(True)

            # plot reference phase vs model phase
            plt.subplot(len_velmean, 2, 2*k+2)
            for i in range(len(forcing)):
                start = len_frq1*i
                end = len_frq1*(i+1)
                plt.plot(frequency, jnp.unwrap(phase_result_array[start : end]), label=f'Model A = {forcing[i]}', marker='o', linewidth=0.8, color=col_vec[i])
                plt.plot(frequency, phase[:, i], label=f'True CFD A = {forcing[i]}', ls='--', marker='x', linewidth=0.8, color=col_vec[i])
            plt.title(f'Comparison Phase: Model vs CFD data || mean vel {vel} m/s')
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Phase [rad]')
            plt.legend()
            plt.grid(True)

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main_FDF_opt1_test()