import matplotlib.pyplot as plt

def loss_plot(loss_history):
    """
    Plots loss history from training on a double log scale
    """
    plt.figure(figsize=(10, 6))
    plt.loglog(loss_history)
    plt.title('Loss History Training')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()


def prediction_plot(time_vec, true, pred, is_nonlinear : bool, is_train : bool):
    """
    Plots time series prediction of model vs reference data
    for const. mean flow velocity
    """
    plt.figure(figsize=(10, 6))
    plt.plot(time_vec, true, label='true')
    plt.plot(time_vec, pred, label='prediction of train data' if is_train else 'predicition of test data', alpha=0.75)
    plt.title('Comparison of nonlinear data' if is_nonlinear else 'Comparison of linear data')
    plt.grid(True)
    plt.xlabel("Time [s]")
    plt.legend()
    plt.show()


def KRNLV_data_plot(KRNLV_DATA, KRNLV_data_set):
    """
    Plots raw Kornilov data

    Parameters
    ----------
    KRNLV_DATA
        Kornilov data set
    KRNLV_data_set
        contains which [perturbation A, time length T_max]
    """
    plt.figure(figsize=(6, 6))
    plt.subplot(2, 1, 1)
    plt.plot(KRNLV_DATA[f'A{KRNLV_data_set[0]}/{KRNLV_data_set[1]}s']['time'], KRNLV_DATA[f'A{KRNLV_data_set[0]}/{KRNLV_data_set[1]}s']['vel'])
    plt.xlabel("Time [s]")
    plt.ylabel("Velocity [m/s]")
    plt.title("Raw Kornilov data: Velocity vs time")
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(KRNLV_DATA[f'A{KRNLV_data_set[0]}/{KRNLV_data_set[1]}s']['time'], KRNLV_DATA[f'A{KRNLV_data_set[0]}/{KRNLV_data_set[1]}s']['hrr'])
    plt.xlabel("Time [s]")
    plt.ylabel("hrr")
    plt.title("Raw Kornilov data: hrr vs time")
    plt.grid()
    plt.tight_layout()
    plt.show()


def KRLNV_mean_plot(time, vel, hrr, vel_mean_slope, hrr_mean_slope, is_train : bool):
    """
    Plots vel and hrr with respective mean slope
    """
    plt.figure(figsize=(6, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time, hrr)
    plt.plot(time, hrr_mean_slope, ls='--')
    plt.xlabel("Time [s]")
    plt.ylabel("hrr")
    plt.title("Train hrr with mean slope" if is_train else "Test hrr with mean slope")
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(time, vel)
    plt.plot(time, vel_mean_slope, ls='--')
    plt.xlabel("Time [s]")
    plt.ylabel("Velocity [m/s]")
    plt.title("Train vel with mean slope" if is_train else "Test vel with mean slope")
    plt.grid()
    plt.tight_layout()
    plt.show()

def KRLNV_norm1_plot(time, vel, hrr, is_train : bool):
    """
    Plots vel and hrr after normalization option 1
    """
    plt.figure(figsize=(6, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time, hrr)
    plt.xlabel("Time [s]")
    plt.ylabel("hrr")
    plt.title("Normalized train hrr vs time" if is_train else "Normalized test hrr vs time")
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(time, vel)
    plt.xlabel("Time [s]")
    plt.ylabel("vel")
    plt.title("Normalized train vel vs time" if is_train else "Normalized test vel vs time")
    plt.grid()
    plt.tight_layout()
    plt.show()