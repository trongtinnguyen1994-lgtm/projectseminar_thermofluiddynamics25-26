# import third party packages
import jax.numpy as jnp
from pathlib import Path
import pickle
import matplotlib.pyplot as plt

# import own packages
from pstfd_project.data_handlers import load_CFD_data, load_data, prep_forcing_signal
from pstfd_project.jax_model_implementation import MyMLP
from pstfd_project.model_functions import model_predict_forc
from pstfd_project.general_utilities import fit_value_jax, get_gain_phase


def main_FDF():

    # ------- CUSTOM PARAMS -------
    # =============================

    # select file
    idx = 1

    # ==========================
    # ------- END CUSTOM -------

    # get project directory
    DIR = Path(__file__).resolve().parent.parent

    # load CFD data
    CFD_DATA = load_CFD_data(DIR / 'Data/CFD_FDF.h5')

    # load nonlinear data
    NL_DATA = load_data(DIR / 'Data/NonlinearFlameData.h5', u_rate=4)
    
    # get all elements in CFD_FDF.h5 file
    #print(CFD_DATA.keys())

    # initialize CFD data
    gain = CFD_DATA['gain']                 # shape(8, 5)
    phase = CFD_DATA['phase']               # shape(8, 5)
    frequency = CFD_DATA['frequency']       # shape(8,)
    forcing = CFD_DATA['forcing']           # shape(5,)

    # check for dimensions
    #print(frequency.shape, gain.shape, phase.shape, forcing.shape)

    print('-------loading model----------')
    # loading pickle file from training
    # path directory to file
    path = Path(DIR / f'Documentation/NoRamp_data/saved_models/model_packages{idx}.pk1')

    # loading file
    with open(path, 'rb') as f:
        package = pickle.load(f)

    # loading data
    params = package['params']
    key = package['key']

    # hyperparmeters
    hyp = package['hyperparams']
    WINDOW_SIZE = hyp['window_size']
    learning_rate = hyp['learning_rate']
    no_neurons_h1 = hyp['no_neurons hidden layer 1']
    no_neurons_h2 = hyp['no_neurons hidden layer 2']
    ORDER = hyp['ORDER']

    # setting up model
    # structure of NN
    shape = [WINDOW_SIZE, no_neurons_h1, no_neurons_h2, 1]

    # creating model
    model = MyMLP(key, shape, learning_rate)

    # set model with saved params
    model.params = params
    
    # lists for model predictions
    gain_result = []
    phase_result = []

    print('-------processing FDF---------')
    for i, frc in enumerate(forcing):
        for j, frq in enumerate(frequency):

            # -------setting up time vector: cutting transient and adjust signal to full periods----------
            dt = NL_DATA['ts'] # =0.0002

            # number of periods of input signal    
            num_periods = 15
            
            # assumed length of transient
            length_transient = 0.015 # [s]

            # setting length of single period
            length_period = 1 / (frq * dt)

            # setting length of whole period
            length_period_steps = int(jnp.round(num_periods * length_period))

            # setting steps of transient
            length_transient_steps = int(jnp.round(length_transient / dt)) # 50 for length_transient=0.015

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
            q_pred = model_predict_forc(model, params, forc_signal, WINDOW_SIZE)
            
            # cut of transient of model prediction
            q_pred_flat = jnp.squeeze(q_pred[length_transient_steps:])

            # get gain / phase of corresponding frequency
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
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for i in range(len(forcing)):
        start = len_frq1*i
        end = len_frq1*(i+1)
        plt.plot(frequency, gain_result_array[start : end], label=f'Model A = {forcing[i]}', color=col_vec[i], marker='o', linewidth=0.8)
        plt.plot(frequency, gain[:, i], label=f'True CFD A = {forcing[i]}', ls='--', color=col_vec[i], marker='o', linewidth=0.8)
    plt.title('Comparison Gain: Model vs CFD data')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Gain [-]')
    plt.legend()
    plt.grid(True)

    # plot reference phase vs model phase
    plt.subplot(1, 2, 2)
    for i in range(len(forcing)):
        start = len_frq1*i
        end = len_frq1*(i+1)
        plt.plot(frequency, jnp.unwrap(phase_result_array[start : end]), label=f'Model A = {forcing[i]}', color=col_vec[i], marker='o', linewidth=0.8)
        plt.plot(frequency, jnp.unwrap(phase[:, i]), label=f'True CFD A = {forcing[i]}', ls='--', color=col_vec[i], marker='o', linewidth=0.8)
    plt.title('Comparison Phase: Model vs CFD data')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Phase [rad]')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    
    # compute + print fit values for gain / phase
    fit_gain = fit_value_jax(gain.T.flatten(), gain_result_array)
    fit_phase = fit_value_jax(phase.T.flatten(), phase_result_array)
    print('\n################################################')
    print(f"Fit value for gain:   {(float(jnp.round(fit_gain, 2))):.2f}%")
    print(f"Fit value for phase:  {(float(jnp.round(fit_phase, 2))):.2f}%")
    print('################################################\n')
    
if __name__ == '__main__':
    main_FDF()