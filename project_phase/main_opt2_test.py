# import third party libraries
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from pathlib import Path
import pickle

# import own packages
from pstfd_project.data_handlers import get_gain_phase_xFDF, load_KRNLV_data, prep_KRNLV_data_opt2, prep_valuation, KRNLV_train_test_split, prep_forcing_signal, load_xFDF_data
from pstfd_project.jax_model_implementation import MyMLP
from pstfd_project.model_functions import model_predict_forc
from pstfd_project.general_utilities import norm2_KRNLV_DATA, get_gain_phase

def main_FDF_opt2_test():

    # ------- CUSTOM PARAMS -------
    # =============================

    # choose model to test
    idx = 1

    # print fit values
    fit_values = False

    # plot model time series prediction
    plot = True

    # plot FDF analysis
    FDF_bool = True

    # FDF in one big figure (vs multiple small figures)
    FDF_oneplot = True

    # set mean velocity for FDF
    vel_mean = [0.4, 0.7]

    # =========================
    # -------END CUSTOM -------

    #-----------prepare Kornilov data set-------------
    # get project directory
    DIR = Path(__file__).resolve().parent.parent

    # load saved models
    # folder directory
    path = DIR / f'Documentation/KRNLV_data/opt2/saved_KRNLV_models_opt2/KRNLV_model_opt2_packages{idx}.pk1'

    if not path.exists():
        raise FileNotFoundError(f'Pickle file for idx: {idx} not found at:\n {path}')

    # load model file
    with open(path, 'rb') as f:
        package = pickle.load(f)

    # loading data
    params = package['params']
    key = package['key']
    shape = package['shape']

    # loading hyperparameters
    hyp = package['hyperparams']
    WINDOW_SIZE = hyp['window_size']
    NO_BATCHES = hyp['no_batches']
    learning_rate = hyp['learning_rate']
    no_neurons_h1 = hyp['no_neurons hidden layer 1']
    no_neurons_h2 = hyp['no_neurons hidden layer 2']
    ORDER = hyp['ORDER']
    UNDERSAMPLING = hyp['undersampling']

    # load Kornilov mean flow data as dictionary
    DIR_KRNLV = DIR / 'Data/Kornilov_Mean_flow_Ramp'
    KRNLV_DATA = load_KRNLV_data(path=DIR_KRNLV, u_rate=UNDERSAMPLING)
    
    # choosing which A for [linear, nonlinear] data
    train_A = [10, 100]
    test_A = [10, 100]

    print('----------normalizing----------')
    # create train / test data (train: T_max=0.2, test: T_max=0.4)
    KRNLV_train, KRNLV_test = KRNLV_train_test_split(KRNLV_DATA, train_A, test_A)

    # normalize data
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
    
    if fit_values:
        # train + test fit value (nonlinear / linear)
        print('\n################################################')
        print(f"Fit value for linear train data: {(float(jnp.round(l_train_fit, 2))):.2f}%")
        print(f"Fit value for linear test data:  {(float(jnp.round(l_test_fit, 2))):.2f}%")
        print('################################################')
        print(f"Fit value for nonlinear train data: {(float(jnp.round(nl_train_fit, 2))):.2f}%")
        print(f"Fit value for nonlinear test data:  {(float(jnp.round(nl_test_fit, 2))):.2f}%")
        print('################################################\n')

    if FDF_bool:
        print('----------FDF analysis---------')
        # post processing for FDF analysis
        
        # load xFDF (Kornilov reference data)
        DIR_xFDF = DIR / 'Data/Kornilov_reference_xFDF'
        xFDF_FULL_DATA = load_xFDF_data(path=DIR_xFDF)

        # throw out harmonic 2, 3 of data set
        xFDF_DATA = [d for d in xFDF_FULL_DATA if d.get('harmonic') != (2, 3)]

        '''
        freq: 100, 170, 200, 300, 400, 500
        amp: 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.5, 0.75, 0.8, 1.0, 1.25, 1.5
        '''
        # set all freq / amps to iterate
        frequency = [25, 50, 75, 100, 170, 200, 300]
        forcing = [0.1, 0.5, 1.5]
        
        # create gain / phase of reference data
        # !structure of gain / phase: array(freq, amps)
        gain, phase = get_gain_phase_xFDF(xFDF_DATA, frequency, forcing)

        # get length of vel_mean for plotting later
        len_velmean = len(vel_mean)

        # one big figure
        if FDF_oneplot:
            plt.figure(figsize=(14, 8))

        """
        u_mean + A * u_mean sin () then norm with norm_fac
        """

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
                    num_periods = 15
                    
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
                    # -- A*u_mean * sin() --
                    forc_signal = prep_forcing_signal(frq, (frc*vel), t_vec)

                    # normalize (u_mean - forc_signal) with saved scalar
                    forc_signal_norm = ((vel + forc_signal) - norm_fac) / norm_fac

                    # predict output on input forcing signal via model
                    q_pred = model_predict_forc(model, params, forc_signal_norm, WINDOW_SIZE)

                    #re-normalize with norm factor
                    q_pred_renorm = q_pred * norm_fac + norm_fac

                    # normalize with mean of hrr
                    hrr_data = KRNLV_DATA['A10/0.2s']['hrr']
                    vel_data = KRNLV_DATA['A10/0.2s']['vel']
                    hrr_norm_fac = jnp.mean(hrr_data[0] * vel/vel_data[0])
                    q_pred_norm = (q_pred_renorm - hrr_norm_fac) / hrr_norm_fac
                    
                    # cut of transient of model prediction
                    q_pred_flat = jnp.squeeze(q_pred_norm[length_transient_steps:])

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
            if FDF_oneplot:
                plt.subplot(len_velmean, 2, 2*k + 1)
            else:
                plt.figure()
            for i in range(len(forcing)):
                start = len_frq1*i
                end = len_frq1*(i+1)
                plt.plot(frequency, gain_result_array[start : end], label=f'Model A = {forcing[i]}', marker='o', linewidth=0.8, color=col_vec[i])
                plt.plot(frequency, gain[:, i], label=f'True CFD A = {forcing[i]}', marker='x', linewidth=0.8, color=col_vec[i], ls='--',)
            plt.title(f'Comparison Gain: Model vs CFD data || mean vel {vel} m/s')
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Gain [-]')
            plt.legend()
            plt.grid(True)

            # plot reference phase vs model phase
            if FDF_oneplot:
                plt.subplot(len_velmean, 2, 2*k + 2)
            else:
                plt.figure()
            for i in range(len(forcing)):
                start = len_frq1*i
                end = len_frq1*(i+1)
                plt.plot(frequency, jnp.unwrap(phase_result_array[start : end]), label=f'Model A = {forcing[i]}', marker='o', linewidth=0.8, color=col_vec[i])
                plt.plot(frequency, phase[:, i], label=f'True CFD A = {forcing[i]}', marker='x', linewidth=0.8, color=col_vec[i], ls='--',)
            plt.title(f'Comparison Phase: Model vs CFD data || mean vel {vel} m/s')
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Phase [rad]')
            plt.legend()
            plt.grid(True)

        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    main_FDF_opt2_test()