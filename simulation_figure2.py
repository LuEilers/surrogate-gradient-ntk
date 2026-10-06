from ntk_definitions import *
from sg_ntk_definitions import *
import warnings
import time
import pickle

def main():
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
    list_training_steps = (0, 1000) # we only consider beginning and end of training
    list_scaling_m = (2, 5, 20)
    list_width_n = (10, 100, 500, 1000)

    # Init key
    key, net_key = random.split(random.PRNGKey(10))

    # Choose surrogate derivative
    sg_string = 'erf' # 'erf' for derivative of erf as surrogate derivative, 'id' for derivative of identity as surrogate derivative
    var_array = [(test, train, circle_middle_x, list_training_steps, n, m, key, net_key) for m in list_scaling_m for n in list_width_n]

    # Calculate analytic SG-NTK
    kernel_ana_list = []
    for m in list_scaling_m:
        shape = (dim, 10, 10, 1)
        kernel_ana_erf = erf_ntk(test_xs, test_xs, len(shape)-1, m, 1, sg_string)
        kernel_ana_list.append(kernel_ana_erf[int(test_points/2),:])

    # Simulate empirical SG-NTK
    start = time.time()
    queue_iterate = [calc_plot_data_sg(test, train, circle_middle_x, list_training_steps, n, m, key, net_key, sg_string) for m in list_scaling_m for n in list_width_n]
    stop = time.time()
    print("Time for computation: ", stop-start, " for n_max =", list_width_n[-1], ", t =", list_training_steps[-1])

    # Save_data
    save_data = [list_width_n, list_scaling_m, list_training_steps, test_xs_1d, queue_iterate, kernel_ana_list]
    with open('data/data_figure2', 'wb') as f:
        pickle.dump(save_data, f)

if __name__ == "__main__":
    main()