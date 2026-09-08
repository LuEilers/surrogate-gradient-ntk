from multiprocess import Pool
from multiprocess.pool import ThreadPool
from ntk_definitions import *
import warnings
import time
import pickle

### CREATE DATA SET

test_points = 100
train_points = 15
target_fn = lambda x: 4*x[0]*x[1]**2 - 0.8*x[0]**3 + 1.2*x[1]**2 - 0.8*x[0]**2*x[1]

test_xs, test_xs_1d, test_ys, train_xs, train_xs_1d, train_ys = generate_dataset(target_fn, train_points, test_points, 0., random.PRNGKey(2))
test = (test_xs, test_ys)
train = (train_xs, train_ys)

circle_middle_x = test_xs[int(test_points/2)]

### DEFINE PARAMETERS

# Parameters
list_training_steps = (0, 10000)
list_scaling_m = (2,5,20)
list_width_n = (10, 100, 500, 1000)

# Init key
key, net_key = random.split(random.PRNGKey(10))

var_array = [(test, train, circle_middle_x, list_training_steps, n, m, key, net_key) for m in list_scaling_m for n in list_width_n]

# Calculate analytic NTKs
kernel_ana_list = []
for m in list_scaling_m:
    shape = (dim, 10, 10, 1)
    init_fn, apply_fn, kernel_fn = stax.serial(
        stax.Dense(shape[1], W_std=sigma_w, b_std=sigma_b), stax.Erf(1,m,0),
        stax.Dense(shape[2], W_std=sigma_w, b_std=sigma_b), stax.Erf(1,m,0),
        stax.Dense(shape[3], W_std=sigma_w, b_std=sigma_b)
    )
    
    apply_fn = jit(apply_fn)
    kernel_fn = jit(kernel_fn, static_argnames='get')
    kernel_ana_list.append(kernel_fn(np.array([circle_middle_x]), test_xs, 'ntk')[0,:])

# Simulate empirical NTKs
start = time.time()
queue_iterate = [calc_plot_data(test, train, circle_middle_x, list_training_steps, n, m, key, net_key) for m in list_scaling_m for n in list_width_n]
stop = time.time()
print("Time for computation: ", stop-start, " for n_max =", list_width_n[-1], ", t =", list_training_steps[-1])

# Save data for plotting
save_data = [list_width_n, list_scaling_m, list_training_steps, test_xs_1d, queue_iterate, kernel_ana_list]
with open('data/data_figure1', 'wb') as f:
    pickle.dump(save_data, f)