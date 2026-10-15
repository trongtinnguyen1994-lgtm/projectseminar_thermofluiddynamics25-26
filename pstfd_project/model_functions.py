import jax
import jax.numpy as jnp
import optax


def train(model, nl_train_in, nl_train_out, l_train_in, l_train_out, epochs, PATIENCE, min_delta, l_error_weight, weights_decay):
        """
        Training loop over all epochs with early stopping
        Implemented train step and loss function

        Parameters
        ----------
        model
            MLP to be trained
        nl_train_in
            nonlinear train input
        nl_train_out
            nonlinear train output
        l_train_in
            linear train input
        l_train_out
            linear train output
        epochs
            number of epochs
        PATIENCE
            number of tolerated epochs without change in error loss
        min_delta
            minimum improvement to restart early stopping counter
        l_error_weight
            weight for linear error in total error
        weights_decay
            L2 regularization rate

        Returns
        -------
        loss_history
            loss over all epochs
        """
        # create a loss history list to store loss values
        params = model.params
        opt_state = model.opt_state
        loss_history = []

        # creating jit-able functions
        model_apply = jax.jit(model.__call__)
        update_fn = model.optimizer.update

        # early stopping
        best_loss = float('inf')
        early_stop_counter = 0

        print("Training...")
        for i_epochs in range(epochs): # training loop --> here a for loop is okay, see the example here: https://keras.io/guides/writing_a_custom_training_loop_in_jax/
            for nl_batch_in, nl_batch_out, l_batch_in, l_batch_out in zip(nl_train_in, nl_train_out, l_train_in, l_train_out):
                # nonlinear
                nl_batch_in = jnp.array(nl_batch_in)
                nl_batch_out = jnp.array(nl_batch_out)

                # linear
                l_batch_in = jnp.array(l_batch_in)
                l_batch_out = jnp.array(l_batch_out)

                params, opt_state = train_step_fn(
                     params, 
                     opt_state, 
                     nl_batch_in, 
                     nl_batch_out,
                     l_batch_in, 
                     l_batch_out, 
                     model_apply,
                     update_fn,
                     l_error_weight,
                     weights_decay
                     ) # update parameters --> the call of the model on x is done as matrix vector multiplication which makes it fast

            # compute loss for first batch
            nl_first_batch_in = jnp.array(nl_train_in[0])
            nl_first_batch_out = jnp.array(nl_train_out[0])

            l_first_batch_in = jnp.array(l_train_in[0])
            l_first_batch_out = jnp.array(l_train_out[0])

            current_loss = loss_fn(
                                params, 
                                nl_first_batch_in, 
                                nl_first_batch_out, 
                                l_first_batch_in, 
                                l_first_batch_out,
                                model_apply
                                ) # check loss for new parameters
            loss_history.append(current_loss) # store loss value for plotting later
            if i_epochs % 5 == 0: 
                print(f"Epoch {i_epochs}: Loss {current_loss:.4f}")

            # early stopping
            if abs(current_loss - best_loss) > min_delta:
                best_loss = current_loss
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            if  early_stop_counter > PATIENCE:
                print('Early stopping after ' + str(i_epochs) + ' epochs.')
                break

        model.params = params
        model.opt_state = opt_state
        return loss_history

#-----------train step--------------

def loss_fn(params, nl_x, nl_y, l_x, l_y, model_apply, l_error_weight=10, weights_decay=1e-2):
        """
        Calculates loss of current training step with Mean Square Error and L2 regularization
        Linear / nonlinear are both weighted (l_error_weight / 1)

        Parameters
        ----------
        params
            Parameters from model
        nl_x
            nonlinear input data
        nl_y
            nonlinear output data
        l_x
            linear input data
        l_y
            linear output data
        model_apply
            function variable for model.__call__
        l_error_weight
            weight for linear error in total error
        weights_decay
            L2 regularization rate

        Returns
        -------
        "current loss"
            = total error + L2 penalty * weights_decay
        """
        nl_prediction = model_apply(params, nl_x)
        l_prediction = model_apply(params, l_x)

        nl_prediction = jnp.squeeze(nl_prediction)  # If jax throws shape errors, check here first --> always squeeze
        l_prediction = jnp.squeeze(l_prediction)

        # MSE
        nl_mse = jnp.mean((nl_prediction - nl_y) ** 2)
        l_mse = jnp.mean((l_prediction - l_y) ** 2)

        # total error with weighted linear error
        total_mse = nl_mse + l_mse * l_error_weight

        # L2 regularization: sum of squared weights
        l2_penalty = sum([jnp.sum(w**2) for w in jax.tree_util.tree_leaves(params)])

        return total_mse + weights_decay * l2_penalty 

# ==========================================
# Training (JIT Compiled) (Must be outside of class)
# ==========================================
@jax.jit(static_argnums=(6, 7)) # 6 & 7th args are static
# this line compiles the function for speed; you can comment it out to see uncompiled version
def train_step_fn(params, opt_state, nl_x, nl_y, l_x, l_y, model_apply, update_fn, l_error_weight, weights_decay): # this is one training step
    """
    one single train step of calculating gradient plus loss with updating parameters of model

    Parameters
    ----------
    params
        Parameters from model
    opt_state
        currrent optimal state from earlier steps
    nl_x
        nonlinear input data
    nl_y
        nonlinear output data
    l_x
        linear input data
    l_y
        linear output data
    model_apply
        function variable for model.__call__
    update_fn
        apply current params to update optimizer (Adam)
    l_error_weight
        weight for linear error in total error
    weights_decay
        L2 regularization rate

    Returns
    -------
    params
    
    opt_state
        new optimal state
    """
    def local_loss(p):
        return loss_fn(p, nl_x, nl_y, l_x, l_y, model_apply, l_error_weight, weights_decay)
    # Calculate gradients of loss function; Loss function receives: params, x, y
    grads = jax.grad(local_loss)(params)
    # Get updates from optimizer
    updates, opt_state = update_fn(grads, opt_state)
    # Apply updates to parameters
    params = optax.apply_updates(params, updates)
    return params, opt_state


def model_predict_forc(model, params, forcing_signal, window_size):
    """
    model prediction for FDF

    Parameters
    ----------
    model
        MLP model
    params
        model parameter
    forcing_signal
        sinoid input signal
    window_size
        data size after slicing

    Returns
    -------
    y_pred
        model prediction
    """
    pred = []
    for i in range(len(forcing_signal) - window_size + 1):
        windows = forcing_signal[i : i+window_size]
        pred.append(windows)
    y_pred = model(params, jnp.array(pred))
    return y_pred


def model_predict_forc_xFDF(model, params, forcing_signal, window_size, vel_mean):
    """
    model prediction for FDF for normalization option 1
    adding mean velocity to every sliced data to the end

    Parameters
    ----------
    model
        MLP model
    params
        model parameter
    forcing_signal
        sinoid input signal
    window_size
        data size after slicing
    vel_mean
        mean flow velocity

    Returns
    -------
    y_pred
        model prediction
    """
    pred = []
    for i in range(len(forcing_signal) - window_size + 1):
        windows = forcing_signal[i : i+window_size]
        pred.append(windows)

    forc_signal = jnp.array(pred)
    # compute array for vel mean
    vel_mean_array = jnp.full((forc_signal.shape[0], 2), vel_mean)

    # add mean velocity to input forcing signal
    forc_in_signal = jnp.concatenate([forc_signal, vel_mean_array], axis=1)

    # predict with NN
    y_pred = model(params, forc_in_signal)
    return y_pred