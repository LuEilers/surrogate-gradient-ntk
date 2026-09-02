import jax
import jax.numpy as np

from jax import random
from jax.example_libraries import optimizers
from jax import jit, grad, vmap

import functools

import neural_tangents as nt
from neural_tangents import stax

from IPython.display import set_matplotlib_formats
import matplotlib_inline

matplotlib_inline.backend_inline.set_matplotlib_formats('pdf', 'svg')
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

import itertools

# Network parameters
dim = 2 # Input dimension

sigma_w = 1
sigma_b = 0.1

ensemble_size = 10

# Generate data set on unit circle
def generate_dataset(target_fn, n_train, n_test, noise_scale, key):
    key, x_key, y_key = random.split(key, 3)
    
    # test points
    test_xs = np.array([[np.sin(2*np.pi * i / n_test - np.pi), np.cos(2*np.pi * i / n_test - np.pi)] for i in range(n_test)])
    test_xs_1d = np.array([2*np.pi * i / n_test - np.pi for i in range(n_test)])
    test_ys = np.array([target_fn(test_xs[i,:]) for i in range(test_xs.shape[0])])
    
    # training points
    random_angles = random.uniform(x_key, (n_train,), minval=-np.pi, maxval=np.pi)
    train_xs_1d = random_angles
    train_xs = np.array([[np.sin(angle), np.cos(angle)] for angle in random_angles])
    train_ys = np.array([target_fn(train_xs[i,:]) for i in range(train_xs.shape[0])])
    train_ys += noise_scale * random.normal(y_key, (n_train,))
    
    return test_xs, test_xs_1d, np.reshape(test_ys, (-1,1)), train_xs, train_xs_1d, np.reshape(train_ys, (-1,1))

# Generate data set on unit circle, sample training points from test points
def generate_dataset_trainfromtest(target_fn, n_train, n_test, noise_scale, key):
    key, x_key, y_key = random.split(key, 3)
    
    # test points
    test_xs = np.array([[np.sin(2*np.pi * i / n_test - np.pi), np.cos(2*np.pi * i / n_test - np.pi)] for i in range(n_test)])
    test_xs_1d = np.array([2*np.pi * i / n_test - np.pi for i in range(n_test)])
    test_ys = np.array([target_fn(test_xs[i,:]) for i in range(test_xs.shape[0])])
    
    # training points
    train_indices = random.choice(x_key, np.arange(n_test), (n_train,), replace=False)
    train_xs_1d = np.array([2*np.pi * i / n_test - np.pi for i in train_indices])
    train_xs = test_xs[train_indices,:]
    train_ys = np.array([target_fn(train_xs[i,:]) for i in range(train_xs.shape[0])])
    train_ys += noise_scale * random.normal(y_key, (n_train,))
    
    return train_indices, test_xs, test_xs_1d, np.reshape(test_ys, (-1,1)), train_xs, train_xs_1d, np.reshape(train_ys, (-1,1))


#def calc_plot_data(axes, ax_errors, subplot_matrix_row, col_index, n, m):
def calc_plot_data(test, train, circle_middle_x, list_training_steps, n, m, key, net_key):
    # Define network
    shape = (dim, n, n, 1)

    init_fn, apply_fn, kernel_fn = stax.serial(
        stax.Dense(shape[1], W_std=sigma_w, b_std=sigma_b), stax.Erf(1,m,0),
        stax.Dense(shape[2], W_std=sigma_w, b_std=sigma_b), stax.Erf(1,m,0),
        stax.Dense(shape[3], W_std=sigma_w, b_std=sigma_b)
    )

    apply_fn = jit(apply_fn)
    kernel_fn = jit(kernel_fn, static_argnames='get')
    
    # Init params (and therefore the network)
    _, params = init_fn(net_key, (-1,dim))
    
    # Learning setup
    learning_rate = 0.1

    opt_init, opt_update, get_params = optimizers.sgd(learning_rate)
    opt_update = jit(opt_update)

    loss = jit(lambda params, x, y: 0.5 * np.mean((apply_fn(params, x) - y) ** 2))
    grad_loss = jit(lambda state, x, y: grad(loss)(get_params(state), x, y))
        
    # Define emp NTK
    emp_ntk_fn = nt.empirical_ntk_fn(apply_fn, vmap_axes=0)
    
    # Define function drawing the emp NTK
    def draw_emp_ntk(key, training_steps):
        train_losses = []
        test_losses = []
        emp_ntk_draw_list = []

        _, params = init_fn(key, (-1, dim))
        opt_state = opt_init(params)

        for i in range(training_steps+1):
            train_losses += [np.reshape(loss(get_params(opt_state), *train), (1,))]
            test_losses += [np.reshape(loss(get_params(opt_state), *test), (1,))]
            if i in list_training_steps:
                emp_ntk_draw_list.append(emp_ntk_fn(np.array([circle_middle_x]), test[0], get_params(opt_state))[0,:])
            
            opt_state = opt_update(i, grad_loss(opt_state, *train), opt_state)

        if training_steps > 0:
            train_losses = np.concatenate(train_losses)
            test_losses = np.concatenate(test_losses)

        return get_params(opt_state), train_losses, test_losses, emp_ntk_draw_list, apply_fn(get_params(opt_state), test[0])
        
    # Plot empirical NTKs
    ensemble_key = random.split(key, ensemble_size)
    _, train_loss, _, emp_ntk_draw_list, apply_fn_list = vmap(draw_emp_ntk, in_axes=(0,None))(ensemble_key, max(list_training_steps))
    
    return train_loss, emp_ntk_draw_list, apply_fn_list