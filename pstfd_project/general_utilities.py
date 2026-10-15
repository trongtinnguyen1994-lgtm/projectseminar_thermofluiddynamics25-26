import numpy as np


def create_windows(input_data, window_size):
    """
    Function to convert a 1D array of inputs into a 2D array containing 
    histories of input data. The window size defines the length of the
    history to be considered.

    Parameters
    ----------
    input_data
        data to be windowed
    window_size

    Returns
    -------
    window
        windowed data
    """
    window = np.zeros([len(input_data), window_size])
    for i in range(len(input_data) - window_size + 1):
        for j in range(window_size):
            window[i, j] = input_data[i+j]
    return window


def create_batches(input_data, output_data, no_batches):
    """
    Takes the provided input and output data and splits it into batches.
    The batches are stored inside a list. The batch size results from the
    number of user specified no_batches. The last batch may be shorter
    (as the length of the data may not be divisible by no_batches).

    Parameters
    ----------
    input_data
        input data to be batched
    output_data
        output data to be batched
    no_batches

    Returns
    -------
    batched_input : np.ndarray, shape(59, no_batches)

    batched_output : list, len(50) of vectors shape(59, 0)
    """
    split_idx = len(output_data) // no_batches
    batched_input = []
    batched_output = []
    for j in range(no_batches-1): # Why only until no_batches-1?
        batched_input.append(input_data[j * split_idx : (j + 1) * split_idx])
        batched_output.append(output_data[j * split_idx : (j + 1) * split_idx])

    # Last batch maybe a bit shorter
    batched_input.append(input_data[(no_batches - 1) * split_idx:])
    batched_output.append(output_data[(no_batches - 1) * split_idx:])
    
    return batched_input, batched_output



def norm1_KRNLV_DATA(train_set, test_set):
    """
    Normalization method 1 with individual mean slope for every data

    Parameters
    ----------
    train_set
        contains train data
    test_set
        contains test data

    Returns
    -------
    l_set_norm
        'train_out': l_train_out_norm,

        'train_in': l_train_in_norm,


        'test_out': l_test_out_norm,

        'test_in': l_test_in_norm,

        'train_time': train_time,

        'test_time': test_time,

        'train_in_mean_slope': train_in_mean_slope,

        'train_out_mean_slope': train_out_mean_slope,

        'test_in_mean_slope': test_in_mean_slope,

        'test_out_mean_slope': test_out_mean_slope

    nl_set_norm
        'train_out': nl_train_out_norm,

        'train_in': nl_train_in_norm,

        'test_out': nl_test_out_norm,

        'test_in': nl_test_in_norm,

        'train_time': train_time,

        'test_time': test_time,

        'train_in_mean_slope': train_in_mean_slope,

        'train_out_mean_slope': train_out_mean_slope,

        'test_in_mean_slope': test_in_mean_slope,

        'test_out_mean_slope': test_out_mean_slope
    """

    # extract all train data
    l_train_in = train_set['l_in']
    l_train_out = train_set['l_out']
    nl_train_in = train_set['nl_in']
    nl_train_out = train_set['nl_out']
    train_time = train_set['time']
    
    # extract all test data
    l_test_in = test_set['l_in']
    l_test_out = test_set['l_out']
    nl_test_in = test_set['nl_in']
    nl_test_out = test_set['nl_out']
    test_time = test_set['time']

    # compute mean slope for linear / nonlinear and train / test
    train_in_mean_slope = 0.3 + (0.8 - 0.3) / (train_time[-1] - train_time[0]) * train_time
    train_out_mean_slope = l_train_out[0] + (l_train_out[0]*0.8/0.3 - l_train_out[0]) / (train_time[-1] - train_time[0]) * train_time
    test_in_mean_slope = 0.3 + (0.8 - 0.3) / (test_time[-1] - test_time[0]) * test_time
    test_out_mean_slope = l_test_out[0] + (l_test_out[0]*0.8/0.3 - l_test_out[0]) / (test_time[-1] - test_time[0]) * test_time

    # normalize all data
    l_train_in_norm = (l_train_in - train_in_mean_slope) / train_in_mean_slope
    l_train_out_norm = (l_train_out - train_out_mean_slope) / train_out_mean_slope
    nl_train_in_norm = (nl_train_in - train_in_mean_slope) / train_in_mean_slope
    nl_train_out_norm = (nl_train_out - train_out_mean_slope) / train_out_mean_slope

    l_test_in_norm = (l_test_in - test_in_mean_slope) / test_in_mean_slope
    l_test_out_norm = (l_test_out - test_out_mean_slope) / test_out_mean_slope
    nl_test_in_norm = (nl_test_in - test_in_mean_slope) / test_in_mean_slope
    nl_test_out_norm = (nl_test_out - test_out_mean_slope) / test_out_mean_slope

    # compress in linear / nonlinear dictionaries
    l_set_norm = {
        'train_out': l_train_out_norm,
        'train_in': l_train_in_norm,
        'test_out': l_test_out_norm,
        'test_in': l_test_in_norm,
        'train_time': train_time,
        'test_time': test_time,
        'train_in_mean_slope': train_in_mean_slope,
        'train_out_mean_slope': train_out_mean_slope,
        'test_in_mean_slope': test_in_mean_slope,
        'test_out_mean_slope': test_out_mean_slope
    }
    nl_set_norm = {
        'train_out': nl_train_out_norm,
        'train_in': nl_train_in_norm,
        'test_out': nl_test_out_norm,
        'test_in': nl_test_in_norm,
        'train_time': train_time,
        'test_time': test_time,
        'train_in_mean_slope': train_in_mean_slope,
        'train_out_mean_slope': train_out_mean_slope,
        'test_in_mean_slope': test_in_mean_slope,
        'test_out_mean_slope': test_out_mean_slope
    }

    return l_set_norm, nl_set_norm


def norm2_KRNLV_DATA(train_set, test_set):
    """
    Normalization method 2 with constant factor from the mean of nonlinear train input data (norm_fac)

    Parameters
    ----------
    train_set
        contains train data
    test_set
        contains test data

    Returns
    -------
    l_set_norm
        'train_out': l_train_out_norm,

        'train_in': l_train_in_norm,

        'test_out': l_test_out_norm,

        'test_in': l_test_in_norm,

        'train_time': train_time,

        'test_time': test_time

    nl_set_norm
        'train_out': nl_train_out_norm,

        'train_in': nl_train_in_norm,

        'test_out': nl_test_out_norm,

        'test_in': nl_test_in_norm,

        'train_time': train_time,

        'test_time': test_time

    norm_fac
        mean of nonlinear train input signal
    """

    # extract all train data
    l_train_in = train_set['l_in']
    l_train_out = train_set['l_out']
    nl_train_in = train_set['nl_in']
    nl_train_out = train_set['nl_out']
    train_time = train_set['time']
    
    # extract all test data
    l_test_in = test_set['l_in']
    l_test_out = test_set['l_out']
    nl_test_in = test_set['nl_in']
    nl_test_out = test_set['nl_out']
    test_time = test_set['time']

    # normalize with mean of nonlinear train data
    norm_fac = jnp.mean(nl_train_in)

    # normalize all data
    l_train_in_norm = (l_train_in - norm_fac) / norm_fac
    l_train_out_norm = (l_train_out - norm_fac) / norm_fac
    nl_train_in_norm = (nl_train_in - norm_fac) / norm_fac
    nl_train_out_norm = (nl_train_out - norm_fac) / norm_fac

    l_test_in_norm = (l_test_in - norm_fac) / norm_fac
    l_test_out_norm = (l_test_out - norm_fac) / norm_fac
    nl_test_in_norm = (nl_test_in - norm_fac) / norm_fac
    nl_test_out_norm = (nl_test_out - norm_fac) / norm_fac

    # compress in linear / nonlinear dictionaries
    l_set_norm = {
        'train_out': l_train_out_norm,
        'train_in': l_train_in_norm,
        'test_out': l_test_out_norm,
        'test_in': l_test_in_norm,
        'train_time': train_time,
        'test_time': test_time
    }
    nl_set_norm = {
        'train_out': nl_train_out_norm,
        'train_in': nl_train_in_norm,
        'test_out': nl_test_out_norm,
        'test_in': nl_test_in_norm,
        'train_time': train_time,
        'test_time': test_time
    }

    return l_set_norm, nl_set_norm, norm_fac


import jax.numpy as jnp

def fit_value_jax(true, model_out):
    """ 
    Returns the %-fit value given a model output signal and a ref. signal

    Parameters
    ----------
    true
        true, measured data

    model_oout
        model prediction

    Returns
    -------
    'fit'
        performace of model prediction
    """
    return float(100 * (1 - jnp.linalg.norm(true - model_out) / jnp.linalg.norm(true - jnp.mean(true))))

def get_gain_phase(signal, dt, freq, forc):
    """
    calculates gain and phase via FTF from model prediction for FDF analysis

    Parameters
    ----------
    signal
        model prediction
    dt
        timestep
    freq
        model prediction for given frequency
    forc
        model prediction for given amplitude

    Returns
    -------
    gain
        extracted gain from complex number
    phase
        extraced phase from complex number
    """

    # computing FFT signal of model ouput
    fft_signal = jnp.fft.rfft(signal)

    # Compute the corresponding frequency vector
    freq_vec = jnp.fft.rfftfreq(len(signal), dt)

    # find idx to nearest freq
    idx = jnp.argmin(jnp.abs(freq_vec - freq))

    # get signal for idx
    fft_signal_at_f = fft_signal[idx]

    # normalize signal of idx
    forc_NN = 2 * jnp.abs(fft_signal_at_f) / len(signal)

    # get phase with added shift
    phase = jnp.angle(fft_signal_at_f) + (jnp.pi / 2)

    # get gain
    gain =  forc_NN / forc
    
    return gain, phase