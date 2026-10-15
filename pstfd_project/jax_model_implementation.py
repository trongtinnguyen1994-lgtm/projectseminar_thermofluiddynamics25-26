import jax
import optax
import jax.numpy as jnp

# ==========================================
# Define Model, Loss and Optimizer
# ==========================================
class MyMLP:
    """
    Multi-Layer Perceptron (MLP) using optimizer Adam
    This class manages initialization of parameters, the forward pass, and optimzer state for 2 hidden layers

    Parameters
    ----------
    layer_sizes
        size of hidden layers
    params
        Parameters of model containing weights and biases
    optimzer
        optax optimizer Adam
    opt_state
        current state of optimizing
    """

    def __init__(self, key, layer_sizes, lr):
        """
        Initialize the MLP with random weights and sets up optimizer Adam

        Parameters
        ----------
        key
            set seed for consistent results
        layer_sizes
            size of hidden layers
        lr
            learning rate
        """
        self.layer_sizes = layer_sizes
        self.params = self.init_params(key, layer_sizes)

        # Optimizer Setup (Optax + Adam)
        self.optimizer = optax.adam(lr)
        self.opt_state = self.optimizer.init(self.params)

    
    # --> recall what we fit here: y = w * x + b
    def init_params(self, key, layer_sizes):
        """
        Creates weights and biases for both hidden layers using normal distribution

        Parameters
        ----------
        key
            set seed for consistent results
        layer_sizes
            size of hidden layers

        Returns
        -------
        params
            Parameters of model
        """
        key1, key2, key3 = jax.random.split(key, 3)
        params = {
            'w1': jax.random.normal(key1, shape=(layer_sizes[0], layer_sizes[1])),
            'b1': jnp.zeros((layer_sizes[1], )),

            'w2': jax.random.normal(key2, shape=(layer_sizes[1], layer_sizes[2])),
            'b2': jnp.zeros((layer_sizes[2], )),

            'w_last': jax.random.normal(key3, shape=(layer_sizes[2], layer_sizes[3])),
        }   
        return params

    # Simple call function --> equivalent to y_pred = model(x)
    def __call__(self, params, x : jnp.ndarray):
        """
        Performs a foward pass through both hidden layers with leaky_relu as activation function

        Parameters
        ----------
        params
            Parameters of model
        x
            input data

        Returns
        -------
        'output'
            prediction of model
        """
        h1 = jax.nn.leaky_relu(x @ params['w1'] + params['b1'], negative_slope=0.1)
        h2 = jax.nn.leaky_relu(h1 @ params['w2'] + params['b2'], negative_slope=0.1)
        return h2 @ params['w_last']
    