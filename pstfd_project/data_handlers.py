import numpy as np
import h5py

def load_data(path, u_rate=1):
    """
    Takes a path to an .h5 file and loads the data from it. Applies the user specified undersampling
    rate and returns the data in a dictionary.

    Parameters
    ----------
    path
        location of .h5 file
    u_rate
        undersampling rate (default=1)

    Returns
    -------
    data_dic : dict
        * 'input_data' : input data
        * 'output_data': output data
        * 'ts' : timesteps
    """
    h5file = h5py.File(path, 'r')   # Defining an h5file object
    input_data = np.array(h5file.get('U'))  # Accessing data called 'U' in this object (our input)
    output_data = np.array(h5file.get('Q')) # Accessing data called 'Q' in this object (our output)
    ts = np.array(h5file.get('Ts')) # Accessing data called 'Ts' in this object (our time step)
    h5file.close() # Close the data file to release memory and avoid clashes when opening sth. else later

    # Undersampling
    input_data = input_data[::u_rate]
    output_data = output_data[::u_rate]
    ts *= u_rate

    # put all data in dict
    data_dic = {'input_data' : input_data, 'output_data' : output_data, 'ts' : ts}
    return data_dic
    

def add_noise(data, noise_level):
    """
    Takes some data argument and add noise to it. The amplitude of the noise is set relative to
    the maximum absolute amplitude of the data.

    Parameters
    ----------
    data
        signal to be added noise
    noise_level

    Returns
    -------
    noisy_data
        signal with noise
    """
    max_noise = np.max(abs(data)) * noise_level
    noise = np.random.uniform(-max_noise,max_noise , data.shape)
    noisy_data = data + noise
    return noisy_data


def load_CFD_data(path):
    """
    Extract all data from file Data/CFD_FDF.h5

    Parameters
    ----------
    path
        path to data CFD_FDF.h5

    Returns
    -------
    data_dict
        'gain': gain,

        'phase': phase,

        'frequency': frequency,

        'forcing': forcing (amplitude)
    """
    h5file = h5py.File(path, 'r')
    gain = np.array(h5file.get('gain'))
    phase = np.array(h5file.get('phase'))
    frequency = np.array(h5file.get('frequency'))
    forcing = np.array(h5file.get('forcing'))
    data_dict = {
        'gain': gain,
        'phase': phase,
        'frequency': frequency,
        'forcing': forcing
    }
    return data_dict


def load_KRNLV_data(path, u_rate):
    """
    loading all Kornilov_Mean_flow_Ramp data in a dicitonary
    Each element contains hrr, time, vel of each amplitude and time vector
    All data have same shape: (100001,) (100001,) (100001,)
    Format of item names: e.g. 'A10/0.2s'

    Parameters
    ----------
    path
        folder of all Kornlilov data
    u_rate
        Undersampling rate of data set

    Returns
    -------
    KRNLV_data
        'hrr': heat release rate,

        'time': time vector

        'vel': velocity,        
    """
    KRNLV_data = {}
    amp = [10, 50, 100]
    t_max = [2, 4]
    for a in amp:
        for t in t_max:
            time = np.load(path / f'Ramp_bb_A{a}_Tmax_0_{t}s/time.npy')
            vel = np.load(path / f'Ramp_bb_A{a}_Tmax_0_{t}s/vel.npy')
            hrr = np.load(path / f'Ramp_bb_A{a}_Tmax_0_{t}s/hrr.npy')
            # undersampling
            time = time[::u_rate]
            vel = vel[::u_rate]
            hrr = hrr[::u_rate]
            # put in dic
            ramp_data = {'hrr' : hrr, 'time' : time, 'vel' : vel}
            KRNLV_data.update({f'A{a}/0.{t}s' : ramp_data})
    return KRNLV_data



import jax.numpy as jnp
import colorednoise as cn

from pstfd_project.general_utilities import create_windows
from pstfd_project.data_handlers import add_noise
from pstfd_project.general_utilities import create_batches, fit_value_jax


def prep_data(
        raw_data, 
        window_size, 
        no_batches,
        brownian_noise_lvl=0.25,
        brownian_noise_factor=0.5,
        white_noise_lvl=0.25, 
        white_noise_factor=0.5,
        ):
    """
    One data set (input, output, time) to be split up into train and test with windowing, batching and noise

    Parameters
    ----------
    raw_data
        data set to be prepared into train / test
    window_size
    no_batches
    brownian_noise_lvl
        default = 0.25
    brownian_noise_factor
        default = 0.5
    white_noise_lvl
        default = 0.25
    white_noise_factor
        default = 0.5

    Returns
    -------
    "train_in"
        batched train input data
    "train_out"
        batched train ouput data
    "test_in"
        test input data
    "test_out"
        test output data
    "ORDER"
        number of history
    "train_time"
        time vector of train data
    "test_time"
        time vector of test data
    """

    # getting nonlinear data
    DATA_IN = raw_data['input_data']
    DATA_OUT = raw_data['output_data']
    DATA_TIME = jnp.arange(0, raw_data['ts'] * len(raw_data['input_data']), raw_data['ts'])

    # train 60% / test 40%
    split_idx = int(len(raw_data['input_data']) * 0.6)
    TRAIN_IN, TEST_IN = jnp.array(DATA_IN[:split_idx]), jnp.array(DATA_IN[split_idx:])
    TRAIN_OUT, TEST_OUT = jnp.array(DATA_OUT[:split_idx]), jnp.array(DATA_OUT[split_idx:])
    TRAIN_TIME, TEST_TIME = jnp.array(DATA_TIME[:split_idx]), jnp.array(DATA_TIME[split_idx:])

    # creating brownian noise
    brownian_noise = cn.powerlaw_psd_gaussian(2, len(TRAIN_OUT), random_state=42)
    brownian_noise = brownian_noise / jnp.max(abs(brownian_noise)) * brownian_noise_lvl * jnp.max(abs(TRAIN_OUT))

    # creating white noise
    white_noise = jnp.array(add_noise(TRAIN_OUT, white_noise_lvl))

    # adding both noises
    TRAIN_OUT += brownian_noise * brownian_noise_factor + white_noise * white_noise_factor

    # Calculate how much history will be needed
    ORDER = jnp.ceil(0.01/raw_data['ts']).astype(int)

    # windows and batching
    TRAIN_IN = create_windows(TRAIN_IN, window_size)
    TEST_IN = create_windows(TEST_IN, window_size)

    TRAIN_IN = TRAIN_IN[:-ORDER]
    TEST_IN = TEST_IN[:-ORDER]

    # adjusting train_time_vec when undersampling
    if len(TRAIN_TIME) > len(TRAIN_IN):
        TRAIN_TIME = TRAIN_TIME[:len(TRAIN_IN)]

    # adjusting test_time_vec when undersampling
    if len(TEST_TIME) > len(TEST_IN):
        TEST_TIME = TEST_TIME[:len(TEST_IN)]

    # cut initial output
    TRAIN_OUT = TRAIN_OUT[ORDER:]
    TEST_OUT = TEST_OUT[ORDER:]

    # batching training data
    batched_train_in, batched_train_out = create_batches(TRAIN_IN, TRAIN_OUT, no_batches)
    batched_train_in = jnp.array(batched_train_in[:-1])
    batched_train_out = jnp.array(batched_train_out[:-1])

    return {
        'train_in': batched_train_in,
        'train_out': batched_train_out,
        'test_in':  jnp.array(TEST_IN),
        'test_out':  jnp.array(TEST_OUT),
        'ORDER': ORDER,
        'train_time': TRAIN_TIME,
        'test_time': TEST_TIME
        }


from pstfd_project.plotting import prediction_plot

def prep_valuation(train_in, train_out, test_in, test_out, train_time, test_time, model, params, plot : bool, is_nonlinear : bool):
    """
    After training the model apply parameters, create prediction and 
    prepare data for evaluation via plotting and fit values

    Parameters
    ----------
    train_in
        train input data
    train_out
        train ouput data
    test_in
        test input data
    test_out
        test output data
    train_time
        train time vector
    test_time
        test time vector
    model
        MLP
    params
        weights and biases of NN
    plot (boolean)
        creates plot
    is_nonlinear
        passed information for plotting

    Returns
    -------
    train_fit
        model performace on training data
    test_fit
        model performance on test data
    prediction_plot (function)
        if plot = True, produces plot of time series prediction
    """

    # setting data up for valuation of training + test data
    train_in_full = jnp.vstack(train_in).astype(jnp.float32)
    train_out_full = jnp.vstack(train_out).astype(jnp.float32)
    test_in_full = jnp.vstack(test_in).astype(jnp.float32)
    test_out_full = jnp.vstack(test_out).astype(jnp.float32)

    # adjusting time vec after vstack
    if len(train_time) > len(train_in_full):
        train_time = train_time[:len(train_in_full)]

    # adjusting test_time_vec when undersampling
    if len(test_time) > len(test_in_full):
        test_time = test_time[:len(test_in_full)]

    # predict
    y_pred_train = model(params, train_in_full)
    y_pred_test = model(params, test_in_full)

    # flatten data
    y_pred_train_flat = jnp.ravel(y_pred_train)
    train_out_flat = jnp.ravel(train_out_full)
    y_pred_test_flat = jnp.ravel(y_pred_test)
    test_out_flat = jnp.ravel(test_out_full)

    # train + test fit value
    train_fit = fit_value_jax(train_out_flat, y_pred_train_flat)
    test_fit = fit_value_jax(test_out_flat, y_pred_test_flat)

    if plot:
    # plotting model prediction on train + test data
        prediction_plot(train_time, train_out_flat, y_pred_train_flat, is_nonlinear, is_train=True) #!! Is time vector necessary??
        prediction_plot(test_time, test_out_flat, y_pred_test_flat, is_nonlinear, is_train=False) #!! Is time vector necessary??

    return train_fit, test_fit


def prep_forcing_signal(frequency, amplitude, t_vec):
    """
    create signal as sinoid with given frequency and amplitude for FDF model prediction

    Parameters
    ----------
    frequency
        given frequency for later prediction
    amplitude
        given amplitude for later prediction
    t_vec
        time vector

    Returns
    -------
    'forcing signal' = amp * sin(2*pi*frq*t_vec) 
    """
    return amplitude * jnp.sin(2 * jnp.pi * frequency * t_vec)


def KRNLV_train_test_split(raw_data, train_set, test_set):
    """
    Divide Kornilov data set into train and test with the criterion: 
    * train with T_max 0.2
    * test with T_max 0.4

    Parameters
    ----------
    raw_data
        raw Kornilov data set
    train_set
        choose [linear , nonlinear] with perturbation A, default=[10,100]
    test_set
        choose [linear , nonlinear] with perturbation A, default=[10,100]

    Returns
    -------
    train_set:
        'l_out': l_train_hrr

        'l_in': l_train_vel

        'nl_out': nl_train_hrr

        'nl_in': nl_train_vel

        'time': train_time

    test_set:
        'l_out': l_test_hrr

        'l_in': l_test_vel

        'nl_out': nl_test_hrr

        'nl_in': nl_test_vel

        'time': test_time
    """

    # extract train data (T_max=0.2s)
    l_train_hrr = raw_data[f'A{train_set[0]}/0.2s']['hrr']
    l_train_vel = raw_data[f'A{train_set[0]}/0.2s']['vel']

    nl_train_hrr = raw_data[f'A{train_set[1]}/0.2s']['hrr']
    nl_train_vel = raw_data[f'A{train_set[1]}/0.2s']['vel']

    train_time = raw_data[f'A{train_set[0]}/0.2s']['time']

    # compute train dictionary
    train_set = {
        'l_out': l_train_hrr,
        'l_in': l_train_vel,
        'nl_out': nl_train_hrr,
        'nl_in': nl_train_vel,
        'time': train_time,
    }

    # extract test data (T_max=0.4s)
    l_test_hrr = raw_data[f'A{test_set[0]}/0.4s']['hrr']
    l_test_vel = raw_data[f'A{test_set[0]}/0.4s']['vel']

    nl_test_hrr = raw_data[f'A{test_set[1]}/0.4s']['hrr']
    nl_test_vel = raw_data[f'A{test_set[1]}/0.4s']['vel']

    test_time = raw_data[f'A{test_set[0]}/0.4s']['time']

    # compute test dictionary
    test_set = {
        'l_out': l_test_hrr,
        'l_in': l_test_vel,
        'nl_out': nl_test_hrr,
        'nl_in': nl_test_vel,
        'time': test_time,
    }

    return train_set, test_set


def prep_KRNLV_data_opt1(
        raw_data,
        window_size, 
        no_batches
        ):
    """
    Prepare data after normalization for training with batching, windowing, noise

    Parameters
    ----------
    raw_data
        normalized linear / nonlinear data set
    window_size
        ''
    no_batches
        ''

    Returns
    -------
    "train_in"
        batched train input data
    "train_out"
        batched train ouput data
    "test_in"
        test input data
    "test_out"
        test output data
    "ORDER"
        number of history
    "train_time"
        time vector of train data
    "test_time"
        time vector of test data
    """

    # getting nonlinear data
    TRAIN_IN = raw_data['train_in']
    TRAIN_OUT = raw_data['train_out']
    TRAIN_TIME = raw_data['train_time']
    TEST_IN = raw_data['test_in']
    TEST_OUT = raw_data['test_out']
    TEST_TIME = raw_data['test_time']
    TRAIN_IN_MEAN_SLOPE = raw_data['train_in_mean_slope']
    TEST_IN_MEAN_SLOPE = raw_data['test_in_mean_slope']

    # calculate timestep (same for train / test)
    ts = TRAIN_TIME[1] - TRAIN_TIME[0]

    # Calculate how much history will be needed
    ORDER = jnp.ceil(0.015/ts).astype(int)

    # windowing all input data (signal + mean)
    TRAIN_IN = create_windows(TRAIN_IN, window_size)
    TEST_IN = create_windows(TEST_IN, window_size)
    TRAIN_IN_MEAN_SLOPE = create_windows(TRAIN_IN_MEAN_SLOPE, window_size)
    TEST_IN_MEAN_SLOPE = create_windows(TEST_IN_MEAN_SLOPE, window_size)

    TRAIN_IN = TRAIN_IN[:-ORDER]
    TEST_IN = TEST_IN[:-ORDER]
    TRAIN_IN_MEAN_SLOPE = TRAIN_IN_MEAN_SLOPE[:-ORDER]
    TEST_IN_MEAN_SLOPE = TEST_IN_MEAN_SLOPE[:-ORDER]

    # adjusting train_time_vec when undersampling
    if len(TRAIN_TIME) > len(TRAIN_IN):
        TRAIN_TIME = TRAIN_TIME[:len(TRAIN_IN)]

    # adjusting test_time_vec when undersampling
    if len(TEST_TIME) > len(TEST_IN):
        TEST_TIME = TEST_TIME[:len(TEST_IN)]

    # cut initial output + time
    TRAIN_OUT = TRAIN_OUT[ORDER:]
    TEST_OUT = TEST_OUT[ORDER:]

    # start / end of mean slope of every window
    TRAIN_IN_MEAN_BC = jnp.stack([TRAIN_IN_MEAN_SLOPE[:, 0], TRAIN_IN_MEAN_SLOPE[:, -1]], axis=1)
    TEST_IN_MEAN_BC = jnp.stack([TEST_IN_MEAN_SLOPE[:, 0], TEST_IN_MEAN_SLOPE[:, -1]], axis=1)

    # adding start / end of mean to every window of signal
    TRAIN_IN = jnp.concatenate([TRAIN_IN, TRAIN_IN_MEAN_BC], axis=1)
    TEST_IN = jnp.concatenate([TEST_IN, TEST_IN_MEAN_BC], axis=1)

    # batching training data
    batched_train_in, batched_train_out = create_batches(TRAIN_IN, TRAIN_OUT, no_batches)
    batched_train_in = jnp.array(batched_train_in[:-1])
    batched_train_out = jnp.array(batched_train_out[:-1])

    return {
        'train_in': batched_train_in,
        'train_out': batched_train_out,
        'test_in':  jnp.array(TEST_IN),
        'test_out':  jnp.array(TEST_OUT),
        'ORDER': ORDER,
        'train_time': TRAIN_TIME,
        'test_time': TEST_TIME
        }


def prep_KRNLV_data_opt2(
        raw_data,
        window_size, 
        no_batches
        ):
    """
    Prepare data after normalization for training with batching, windowing, noise

    Parameters
    ----------
    raw_data
        normalized linear / nonlinear data set
    window_size
        ''
    no_batches
        ''

    Returns
    -------
    "train_in"
        batched train input data
    "train_out"
        batched train ouput data
    "test_in"
        test input data
    "test_out"
        test output data
    "ORDER"
        number of history
    "train_time"
        time vector of train data
    "test_time"
        time vector of test data
    """

    # getting nonlinear data
    TRAIN_IN = raw_data['train_in']
    TRAIN_OUT = raw_data['train_out']
    TRAIN_TIME = raw_data['train_time']
    TEST_IN = raw_data['test_in']
    TEST_OUT = raw_data['test_out']
    TEST_TIME = raw_data['test_time']

    # calculate timestep (same for train / test)
    ts = TRAIN_TIME[1] - TRAIN_TIME[0]

    # Calculate how much history will be needed
    ORDER = jnp.ceil(0.015/ts).astype(int)

    # windowing all input data (signal + mean)
    TRAIN_IN = create_windows(TRAIN_IN, window_size)
    TEST_IN = create_windows(TEST_IN, window_size)

    TRAIN_IN = TRAIN_IN[:-ORDER]
    TEST_IN = TEST_IN[:-ORDER]

    # adjusting train_time_vec when undersampling
    if len(TRAIN_TIME) > len(TRAIN_IN):
        TRAIN_TIME = TRAIN_TIME[:len(TRAIN_IN)]

    # adjusting test_time_vec when undersampling
    if len(TEST_TIME) > len(TEST_IN):
        TEST_TIME = TEST_TIME[:len(TEST_IN)]

    # cut initial output + time
    TRAIN_OUT = TRAIN_OUT[ORDER:]
    TEST_OUT = TEST_OUT[ORDER:]

    # batching training data
    batched_train_in, batched_train_out = create_batches(TRAIN_IN, TRAIN_OUT, no_batches)
    batched_train_in = jnp.array(batched_train_in[:-1])
    batched_train_out = jnp.array(batched_train_out[:-1])

    return {
        'train_in': batched_train_in,
        'train_out': batched_train_out,
        'test_in':  jnp.array(TEST_IN),
        'test_out':  jnp.array(TEST_OUT),
        'ORDER': ORDER,
        'train_time': TRAIN_TIME,
        'test_time': TEST_TIME
        }

import json

def load_xFDF_data(path):
    """
    Get all gain and phase values from Data/Kornilov_reference_xFDF

    Parameters
    ----------
    path
        ''

    Returns
    -------
    xFDF_data
        'folder': foldername

        'mean_velocity': mean flow velocity

        'frequency': frequency

        'amplitude': amplitude

        'harmonic': harmonic

        'gain': gain value

        'phase': phase value
    """

    # initialise result list
    xFDF_data = []

    # loop for 0.4 / 0.7 mean
    bases = [4, 7] # !DON'T CHANGE (look into Kornilov reference data set)

    # extract all desired data of all folder
    for b in bases:
        # get file path
        file = path/f'Kornilov_xFDF_uncertain_base0{b} 1.json'

        # open json file
        with open(file, 'r') as f:
            data = json.load(f)
            
        for entry in data:
            freq = entry.get('frequency')
            amp = entry.get('amplitude')

            for h in entry.get('harmonics', []):
                harmonic = h.get('harmonic')
                gain = h.get('gain')
                phase = h.get('phase')

                xFDF_data.append({
                    'folder': entry.get('folder'),
                    'mean_velocity': b,
                    'frequency': freq,
                    'amplitude': amp,
                    'harmonic': harmonic,
                    'gain': gain,
                    'phase': phase
                })

    return xFDF_data


def get_gain_phase_xFDF(data, freq_target, amp_target):
    """
    extract gain / phase for desired frequency and amplitude out of xFDF data
    (part of function: load_xFDF_DATA)

    Parameters
    ----------
    data
        ''
    freq_target
        frequencies to be selected
    amp_target
        amplitudes to be selected

    Returns
    -------
    gain_grid : jnp.array
        array of gain values
    phase_grid : jnp.array
        phase of gain values
    """
    # create 2D list
    gain_grid = [[jnp.nan for _ in amp_target] for _ in freq_target]
    phase_grid = [[0.0 for _ in amp_target] for _ in freq_target]

    # get gain / phase data for given frq / amp
    for i, frq in enumerate(freq_target):
        for j, amp in enumerate(amp_target):
            for entry in data:
                if entry['frequency'] == frq and entry['amplitude'] == amp:
                    gain_grid[i][j] = entry['gain']
                    phase_grid[i][j] = entry['phase']
                    break #end when found

    # HARDCODING: shift phase of amp=.1 & freq=300 down by 2pi
    phase_grid = jnp.array(phase_grid)
    phase_grid = jnp.unwrap(phase_grid, axis=0)
    phase_grid = phase_grid.at[-1, 0].add(-2*jnp.pi)
                 
    return jnp.array(gain_grid), jnp.array(phase_grid)