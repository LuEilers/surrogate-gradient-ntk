from ntk_definitions import *
from sg_ntk_definitions import *
import warnings
import time
import pickle

def main():
    ### CREATE DATA SET

    test_points = 100
    train_points = 15

    # TARGET FUN
    target_fn = lambda x: 4*x[0]*x[1]**2 - 0.8*x[0]**3 + 1.2*x[1]**2 - 0.8*x[0]**2*x[1]

    train_indices, test_xs, test_xs_1d, test_ys, train_xs, train_xs_1d, train_ys = generate_dataset_trainfromtest(target_fn, train_points, test_points, 0., random.PRNGKey(3))
    test = (test_xs, test_ys)
    train = (train_xs, train_ys)

    # Init key
    key, net_key = random.split(random.PRNGKey(10))

    # Parameters
    training_steps = 30000

    # Should be changed to n=20,100,500 to generate all data necessary for Figure 3b
    n = 500
    kappa = 1 # kappa=1 by default, see kappa=0.2 in appendix
    sg_string = 'erf' # 'erf' for derivative of erf as surrogate derivative, 'id' for derivative of identity as surrogate derivative
    ensemble_size = 500

    # Simulate SGL, and calculate SG-NTK (mean and confidence band) & NNGP (i.e., \Sigma_\sign)
    start = time.time()
    train_loss, apply_fn_list, mu, mu_nngp, std = calc_plot_sgl_vs_sgntk(test, train, train_indices, training_steps, n, key, net_key, sg_string, kappa, ensemble_size)
    stop = time.time()
    print("Time for computation: ", stop-start, " for n =", n, ", t =", training_steps)

    # Save data
    save_data = [test_xs_1d, train_xs_1d, train_ys, train_loss, apply_fn_list, mu, mu_nngp, std]
    with open(f'data/data_n{n}_figure3', 'wb') as f:
        pickle.dump(save_data, f)

if __name__ == "__main__":
    main()