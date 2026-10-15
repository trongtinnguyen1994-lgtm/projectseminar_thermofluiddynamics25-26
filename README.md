# **Project Seminar**<br>Deep Learning Strategies for Modeling Hydrogen Combustion
**Author:**             Trong Tin Nguyen, 2025/26  
**Version:**            1.0.0 (April 2026)  <br>    
Supervised by Gregor Doehner, M.Sc.  
Thermo-Fluid Dynamics Group  
TUM School of Engineering and Design  
Technical University of Munich  

## Description
This project seminar aims to analyze the performance of **Neural Networks (NN)** in flame dynamics using a **Multi-Layer Perceptron (MLP)**. The challenge here is to utilize data with linearly ramped-up mean flow velocity, rather than a constant mean velocity, to extract information across different flame types.
This repository consists of an MLP for both constant and ramped-up data, alongside an additional **Flame Describing Function (FDF)** analysis. To train on ramped-up data **(Kornilov data)** the project tested two normalization options **(Opt1 & Opt2)**. The overall objective is to validate NNs for flame analysis and directly compare both normalization methods. This project is working with JAX. For further information on this topic refer to the respecive report of this Project Seminar or the research of the chair [Thermo-Fluid Dynamics](https://www.epc.ed.tum.de/tfd/startseite/) of TU Munich .

## Setup
*   Create **.venv** (Python Version 3.11.0 recommended) 
*   Enter in terminal to install requirements:

    ```bash
    pip install -e .
    ```

## Structure
* **Data**:                                 flame data from experiments
    * **AddData**:                          Additional data with const. mean velocity (here for validation)
    * **Kornilov_Mean_flow_Ramp**:          data set with linearly ramped-up mean flow velocity
        - Input:    flame velocity
        - Output:   heat release rate
        - Time vector

    * **Kornilov_reference_xFDF**:          reference data for FDF of Kornilov data
    * **CFD_FDF.h5**:                       reference data for FDF of const. mean
    * **LinearFlameData.h5**:               linear training data set (const.)
    * **NonlinearFlameData.h5**:            nonlinear training data set (const.)

* **Documentation**:                        saved Neural Network (NN) models with hyperparameters
    * **KRNLV_data**:                       saved models and hyperparameters of MLP on Kornilov data
        * **Opt1 / Opt2**:                  pickle files for models + txt.file of hyperparameters

* **Figures**:                              saved plots
    * **Best_Models**:                      best models from project phase
        - NoRamp (Const.)
        - Opt1
        - Opt2

* **main**:                                 files to run for training routine + FDF
    * **MLP**:                              MLP train + MLP test + FDF with const. mean velocity
    * **Opt1**:                             Opt1 train + Opt1 test on Kornilov data (FDF implemented in test)
    * **Opt2**:                             Opt2 train + Opt2 test on Kornilov data (FDF implemented in test)

* **pstfd_project**:                        own package full of functions
    * **__init__.py**:                      Necessary empty file for pip
    * **data_handlers**:                    loading data, preparing for training
    * **general_utilities**:                windowing, batching, normalization, fit values
    * **jax_model_implementation**:         MLP class
    * **model_functions**:                  training loop, training step, functions for FDF
    * **plotting**:                         plotting for loss and all time series prediction
    * **save_hyperparams**:                 saving hyperparams (.txt) and model (.pkl)

* **README.md**

* **setup.py**:                     for pip to install requirements

## Usage Guide
Most important folders for usage are **main** and **Documentation**.

In **main**, there is a general structure of:
* **train**: adjust hyperparameters and start training to create MLP model
* **test**: load existing model and re-run training, plot time series prediction / fit values
* **FDF**: load existing model and run a FDF analysis (for Opt1 & 2 included in test file)  

In every file, the main() function starts with a **--- CUSTOM PARAMS ---** section where all (hyper-)parameters / variables for the training routine, plotting and saving can be set individually. After **--- END CUSTOM ---** there is no need to change the code.


Saved models are stored in **Documentation**. The respective .txt file displays an overview of all parameters and fit values. When deleting files of model / parameters, it is recommended to remove both model and corresponding text section to keep the intended structure.

## Data
Gereral info to Kornilov data set:
* Perturbation A:      10, 50, 100
* length time T_max:  0.2, 0.4
* A:10 linear, A:50 nonlinear, A:100 highly nonlinear
* velocity = input, heat release rate (hrr) = output
* both to be normalized with both methods
* mean velocity data runs from 0.3 to 0.8
* hrr data runs from 0.2 to 0.52

Approach: 
* train with T_max=0.2
* test with T_max=0.4
* linear data A=10
* nonlinear data A=100
* ignore A=50

## Known Issues / Possible Improvements
* **Phase Shifting**:  
When generating FDF plot for both options, the phase for last frequency may be shifted up by 2*pi, possibly due to unfortunate structure of phase result array combined with jnp.unwrap (e.g Opt1, mean vel @ 0.3 m/s)
* **Reference Gap**:  
xFDF reference data only for 0.4 / 0.7 m/s --> Comparison for other mean flow velocities not representative
* **Low-Frequency Handling**:  
Due to missing reference data in xFDF for low frequencies, those are set 0.0 for phase, while it is possible for gain to be jnp.nan. Find a way to set also phase for lower frequencies to None / jnp.nan to get rid of those points in the FDF plot.