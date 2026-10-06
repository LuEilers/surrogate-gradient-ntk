import jax
import jax.numpy as jnp
import numpy as np

from jax import random
from jax.example_libraries import optimizers
from jax import jit, grad, vmap

import functools

import neural_tangents as nt
from neural_tangents import stax

import itertools

import operator as op
from typing import Callable, Optional, Sequence
import warnings

from jax import custom_jvp, grad, vmap
from jax.scipy.special import erf
from neural_tangents._src.stax.requirements import Diagonal, get_diagonal, get_diagonal_outer_prods, layer, requires, supports_masking
import scipy as sp
from neural_tangents._src.utils import utils
from neural_tangents._src.utils.kernel import Kernel
from neural_tangents._src.utils.typing import InternalLayer, LayerKernelFn

from neural_tangents._src.stax.elementwise import _elementwise

#################################################################
# 1 Define custom activation function with surrogate derivative #
#################################################################

surr_act_erf = erf
surr_act_id = lambda x:x

@stax.layer
@stax.supports_masking(remask_kernel=True)
def surr_Erf(
    a: float = 1.,
    b: float = 1.,
    c: float = 0.
) -> InternalLayer:
  """Affine transform of `Erf` nonlinearity, i.e. `a * Erf(b * x) + c`.

  Args:
    a: output scale.
    b: input scale.
    c: output shift.

  Returns:
    `(init_fn, apply_fn, kernel_fn)`.
  """
  def fn(x):
    return surr_act_erf(x) - jax.lax.stop_gradient(surr_act_erf(x)) + jax.lax.stop_gradient(a * erf(b * x) + c)

  def kernel_fn(k: Kernel) -> Kernel:
    k *= b

    cov1, nngp, cov2, ntk = k.cov1, k.nngp, k.cov2, k.ntk

    cov1_denom = 1 + 2 * cov1
    cov2_denom = None if cov2 is None else 1 + 2 * cov2

    prod11, prod12, prod22 = get_diagonal_outer_prods(cov1_denom,
                                                      cov2_denom,
                                                      k.diagonal_batch,
                                                      k.diagonal_spatial,
                                                      op.mul)

    factor = 2 / jnp.pi

    def nngp_ntk_fn(
        nngp: jnp.ndarray,
        prod: jnp.ndarray,
        ntk: Optional[jnp.ndarray] = None
    ) -> tuple[jnp.ndarray, Optional[jnp.ndarray]]:
      square_root = _sqrt(prod - 4 * nngp**2)
      nngp = factor * jnp.arctan2(2 * nngp, square_root)

      if ntk is not None:
        dot_sigma = 2 * factor / square_root
        ntk *= dot_sigma

      return nngp, ntk

    def nngp_fn_diag(nngp: jnp.ndarray) -> jnp.ndarray:
      return factor * jnp.arctan2(nngp, jnp.sqrt(nngp + 1. / 4))

    nngp, ntk = nngp_ntk_fn(nngp, prod12, ntk)

    if k.diagonal_batch and k.diagonal_spatial:
      cov1 = nngp_fn_diag(cov1)
      if cov2 is not None:
        cov2 = nngp_fn_diag(cov2)
    else:
      cov1, _ = nngp_ntk_fn(cov1, prod11)
      if cov2 is not None:
        cov2, _ = nngp_ntk_fn(cov2, prod22)

    k = k.replace(cov1=cov1, nngp=nngp, cov2=cov2, ntk=ntk)
    return a * k + c

  return _elementwise(fn, f'Erf({a}, {b}, {c})', kernel_fn)

@stax.layer
@stax.supports_masking(remask_kernel=True)
def surr_Id(
    a: float = 1.,
    b: float = 1.,
    c: float = 0.
) -> InternalLayer:
  """Affine transform of `Erf` nonlinearity, i.e. `a * Erf(b * x) + c`.

  Args:
    a: output scale.
    b: input scale.
    c: output shift.

  Returns:
    `(init_fn, apply_fn, kernel_fn)`.
  """
  def fn(x):
    return surr_act_id(x) - jax.lax.stop_gradient(surr_act_id(x)) + jax.lax.stop_gradient(a * erf(b * x) + c)

  def kernel_fn(k: Kernel) -> Kernel:
    k *= b

    cov1, nngp, cov2, ntk = k.cov1, k.nngp, k.cov2, k.ntk

    cov1_denom = 1 + 2 * cov1
    cov2_denom = None if cov2 is None else 1 + 2 * cov2

    prod11, prod12, prod22 = get_diagonal_outer_prods(cov1_denom,
                                                      cov2_denom,
                                                      k.diagonal_batch,
                                                      k.diagonal_spatial,
                                                      op.mul)

    factor = 2 / jnp.pi

    def nngp_ntk_fn(
        nngp: jnp.ndarray,
        prod: jnp.ndarray,
        ntk: Optional[jnp.ndarray] = None
    ) -> tuple[jnp.ndarray, Optional[jnp.ndarray]]:
      square_root = _sqrt(prod - 4 * nngp**2)
      nngp = factor * jnp.arctan2(2 * nngp, square_root)

      if ntk is not None:
        dot_sigma = 2 * factor / square_root
        ntk *= dot_sigma

      return nngp, ntk

    def nngp_fn_diag(nngp: jnp.ndarray) -> jnp.ndarray:
      return factor * jnp.arctan2(nngp, jnp.sqrt(nngp + 1. / 4))

    nngp, ntk = nngp_ntk_fn(nngp, prod12, ntk)

    if k.diagonal_batch and k.diagonal_spatial:
      cov1 = nngp_fn_diag(cov1)
      if cov2 is not None:
        cov2 = nngp_fn_diag(cov2)
    else:
      cov1, _ = nngp_ntk_fn(cov1, prod11)
      if cov2 is not None:
        cov2, _ = nngp_ntk_fn(cov2, prod22)

    k = k.replace(cov1=cov1, nngp=nngp, cov2=cov2, ntk=ntk)
    return a * k + c

  return _elementwise(fn, f'Erf({a}, {b}, {c})', kernel_fn)

@layer
@supports_masking(remask_kernel=False)
def surr_Sign_Erf() -> InternalLayer:
  """Sign function.

  Returns:
    `(init_fn, apply_fn, kernel_fn)`.
  """

  def fn(x):
    return surr_act_erf(x) - jax.lax.stop_gradient(surr_act_erf(x)) + jax.lax.stop_gradient(jnp.sign(x))

  def kernel_fn(k: Kernel) -> Kernel:
    cov1, nngp, cov2, ntk = k.cov1, k.nngp, k.cov2, k.ntk
    if ntk is not None:
      ntk = np.zeros_like(ntk)
    _, prod12, _ = get_diagonal_outer_prods(cov1,
                                            cov2,
                                            k.diagonal_batch,
                                            k.diagonal_spatial,
                                            op.mul)
    angles = _arctan2(_sqrt(prod12 - nngp**2), nngp, fill_zero=np.pi / 2)
    nngp = 1 -  angles * 2 / np.pi
    cov1 = jnp.where(cov1 == 0., 0., 1.)
    cov2 = cov2 if cov2 is None else jnp.where(cov2 == 0, 0., 1.)
    k = k.replace(cov1=cov1, nngp=nngp, cov2=cov2, ntk=ntk)
    return k

  return _elementwise(fn, 'surr_Sign_Erf', kernel_fn)

@layer
@supports_masking(remask_kernel=False)
def surr_Sign_Id() -> InternalLayer:
  """Sign function.

  Returns:
    `(init_fn, apply_fn, kernel_fn)`.
  """

  def fn(x):
    return surr_act_id(x) - jax.lax.stop_gradient(surr_act_id(x)) + jax.lax.stop_gradient(jnp.sign(x))

  def kernel_fn(k: Kernel) -> Kernel:
    cov1, nngp, cov2, ntk = k.cov1, k.nngp, k.cov2, k.ntk
    if ntk is not None:
      ntk = np.zeros_like(ntk)
    _, prod12, _ = get_diagonal_outer_prods(cov1,
                                            cov2,
                                            k.diagonal_batch,
                                            k.diagonal_spatial,
                                            op.mul)
    angles = _arctan2(_sqrt(prod12 - nngp**2), nngp, fill_zero=np.pi / 2)
    nngp = 1 -  angles * 2 / np.pi
    cov1 = jnp.where(cov1 == 0., 0., 1.)
    cov2 = cov2 if cov2 is None else jnp.where(cov2 == 0, 0., 1.)
    k = k.replace(cov1=cov1, nngp=nngp, cov2=cov2, ntk=ntk)
    return k

  return _elementwise(fn, 'surr_Sign_Id', kernel_fn)

@functools.partial(custom_jvp, nondiff_argnums=(2,))
def _arctan2(x, y, fill_zero: Optional[float] = None):
  if fill_zero is not None:
    return jnp.where(jnp.bitwise_and(x == 0., y == 0.),
                    fill_zero,
                    jnp.arctan2(x, y))
  return jnp.arctan2(x, y)

@getattr(_arctan2, 'defjvp', lambda f: f)  # Equivalent to `@_arctan2.defjvp`.
def _arctan2_jvp(fill_zero, primals, tangents):
  x, y = primals
  x_dot, y_dot = tangents
  primal_out = _arctan2(x, y, fill_zero)
  safe_tol = 1e-30
  denom = jnp.maximum(x**2 + y**2, safe_tol)
  tangent_out = x_dot * (y / denom) - y_dot * (x / denom)
  return primal_out, tangent_out

@functools.partial(custom_jvp, nondiff_argnums=(2,))
def _arctan2(x, y, fill_zero: Optional[float] = None):
  if fill_zero is not None:
    return jnp.where(jnp.bitwise_and(x == 0., y == 0.),
                    fill_zero,
                    jnp.arctan2(x, y))
  return jnp.arctan2(x, y)

@getattr(_arctan2, 'defjvp', lambda f: f)  # Equivalent to `@_arctan2.defjvp`.
def _arctan2_jvp(fill_zero, primals, tangents):
  x, y = primals
  x_dot, y_dot = tangents
  primal_out = _arctan2(x, y, fill_zero)
  safe_tol = 1e-30
  denom = jnp.maximum(x**2 + y**2, safe_tol)
  tangent_out = x_dot * (y / denom) - y_dot * (x / denom)
  return primal_out, tangent_out

@functools.partial(custom_jvp, nondiff_argnums=(1,))
def _sqrt(x, tol=0.):
  return jnp.sqrt(jnp.maximum(x, tol))

@getattr(_sqrt, 'defjvp', lambda f: f)  # ReadTheDocs-friendly `@_sqrt.defjvp`.
def _sqrt_jvp(tol, primals, tangents):
  x, = primals
  x_dot, = tangents
  safe_tol = max(tol, 1e-30)
  square_root = _sqrt(x, safe_tol)
  square_root_out = _sqrt(x, tol)
  return square_root_out, jnp.where(x > safe_tol, x_dot / (2 * square_root), 0.)

##########################################
# 2 Define functions for analytic SG-NTK #
##########################################

def arcsin_kernel(x,y,z,m):
    # Identify m = -1 with m = inf
    if m == -1:
        return 2 * sigma_w**2 / np.pi * np.arcsin(z/np.sqrt(x*y)) + sigma_b**2
    else:
        return 2 * sigma_w**2 / np.pi * np.arcsin(z/np.sqrt((1/(2*m**2)+x)*(1/(2*m**2)+y))) + sigma_b**2

def det_kernel(x,y,z,m1,m2,surr):
    if surr == 'erf':
        if m1 == -1:
            if m2 == -1:
                if np.abs(x*y - z**2) < 1e-30:
                    return np.nan
                else:
                    return 2 * sigma_w**2 / np.pi * (x*y - z**2)**(-1/2)
            else:
                return 2 * sigma_w**2 / np.pi * (x*(1/(2*m2**2) + y) - z**2)**(-1/2)
        else:
            return 2 * sigma_w**2 / np.pi * ((1/(2*m1**2) + x)*(1/(2*m2**2) + y) - z**2)**(-1/2)
    elif surr == 'id':
        if m1 == -1:
            return sigma_w**2 * np.sqrt(2/np.pi) * (x)**(-1/2)
        else:
            return sigma_w**2 * np.sqrt(2/np.pi) * (1/(2*m1**2) + x)**(-1/2)

def erf_nngp(z1_array, z2_array, depth, m=1):
    z1_len = z1_array.shape[0]
    z2_len = z2_array.shape[0]
    nngp_array = []
    for i in range(depth):
        if i == 0:
            M = np.array([sigma_w**2 / dim * np.dot(z1,z2) + sigma_b**2 for z1 in z1_array for z2 in z2_array])
            M = M.reshape((z1_len, z2_len))
            nngp_array.append(M)
        else:
            N = nngp_array[i-1]
            M = np.array([arcsin_kernel(N[i,i], N[j,j], N[i,j],m) for i in range(z1_len) for j in range(z2_len)])
            M = M.reshape((z1_len, z2_len))
            nngp_array.append(M)
    return nngp_array

def erf_sigma_dot(N, m1, m2, surr):
    dim1 = N.shape[0]
    dim2 = N.shape[1]
    M = np.array([det_kernel(N[i,i],N[j,j],N[i,j],m1,m2,surr) for i in range(dim1) for j in range(dim2)])
    M = M.reshape((dim1,dim2))
    return M

def erf_ntk(z1_array, z2_array, depth, m1, m2, surr):
    nngp_array = erf_nngp(z1_array, z2_array, depth, m1)
    sigma_dot_array = [erf_sigma_dot(nngp_array[i],m1,m2,surr) for i in range(depth-1)]
    for i in range(depth):
        if i == 0:
            ntk = nngp_array[i]
        else:
            ntk = np.multiply(ntk, sigma_dot_array[i-1]) + nngp_array[i]
    return ntk

####################################
# 3 Simulation and data generation #
####################################

# Network parameters
dim = 2 # Input dimension

sigma_w = 1
sigma_b = 0.1

ensemble_size = 10

# Generate data for plotting empirical SG-NTKs and training/test losses
def calc_plot_data_sg(test, train, circle_middle_x, list_training_steps, n, m, key, net_key, sg_string):

    # Define network
    shape = (dim, n, n, 1)

    init_fn_erf, apply_fn_erf, kernel_fn_erf = stax.serial(
        stax.Dense(shape[1], W_std=sigma_w, b_std=sigma_b), stax.Erf(1,m,0), 
        stax.Dense(shape[2], W_std=sigma_w, b_std=sigma_b), stax.Erf(1,m,0), 
        stax.Dense(shape[3], W_std=sigma_w, b_std=sigma_b)
    )

    if sg_string == 'erf':
        surr_fun = surr_Erf
    else:
        surr_fun = surr_Id
        
    init_fn_surr_erf, apply_fn_surr_erf, kernel_fn_surr_erf = stax.serial(
        stax.Dense(shape[1], W_std=sigma_w, b_std=sigma_b), surr_fun(1,m,0), 
        stax.Dense(shape[2], W_std=sigma_w, b_std=sigma_b), surr_fun(1,m,0), 
        stax.Dense(shape[3], W_std=sigma_w, b_std=sigma_b)
    )

    apply_fn_erf = jit(apply_fn_erf)
    apply_fn_surr_erf = jit(apply_fn_surr_erf)
    kernel_fn_erf = jit(kernel_fn_erf, static_argnames='get')
    kernel_fn_surr_erf = jit(kernel_fn_surr_erf, static_argnames='get')

    # Init params (and therefore the network)
    _, params = init_fn_erf(net_key, (-1,dim))
    
    # Learning setup
    learning_rate = 0.1

    opt_init, opt_update, get_params = optimizers.sgd(learning_rate)
    opt_update = jit(opt_update)

    loss = jit(lambda params, x, y: 0.5 * np.mean((apply_fn_surr_erf(params, x) - y) ** 2))
    grad_loss = jit(lambda state, x, y: grad(loss)(get_params(state), x, y))

    # 1
    eval_apply_fn_erf = jit(lambda params, x: apply_fn_erf(params, x)[0,0])
    jacobi_erf = jit(lambda params, x: grad(eval_apply_fn_erf)(params, x))
    # 2
    eval_apply_fn_surr_erf = jit(lambda params, x: apply_fn_surr_erf(params, x)[0,0])
    jacobi_surr_erf = jit(lambda params, x: grad(eval_apply_fn_surr_erf)(params, x))

    # Define emp SG-NTK
    def emp_sg_ntk(params_):
        vec_jacobi_surr_erf = jnp.concatenate(list(sum(jacobi_surr_erf(params_, circle_middle_x), ())), axis=None)
        vec_jacobi_erf = jnp.array([jnp.concatenate(list(sum(jacobi_erf(params_, test[0][i,:]), ())), axis=None) for i in range(test[0].shape[0])])
        return vec_jacobi_surr_erf @ vec_jacobi_erf.T
    
    # Define function drawing the emp NTK
    def draw_emp_sg_ntk(key, training_steps):
        train_losses = []
        test_losses = []
        emp_sg_ntk_draw_list = []

        _, params = init_fn_erf(key, (-1, dim))
        opt_state = opt_init(params)

        for i in range(training_steps+1):
            train_losses += [np.reshape(loss(get_params(opt_state), *train), (1,))]
            test_losses += [np.reshape(loss(get_params(opt_state), *test), (1,))]
            if i in list_training_steps:
                emp_sg_ntk_draw_list.append(emp_sg_ntk(get_params(opt_state)))
            if i < training_steps:
              opt_state = opt_update(i, grad_loss(opt_state, *train), opt_state)

        if training_steps > 0:
            train_losses = jnp.concatenate(train_losses)
            test_losses = jnp.concatenate(test_losses)

        return get_params(opt_state), train_losses, test_losses, emp_sg_ntk_draw_list, apply_fn_erf(get_params(opt_state), test[0])

    # Plot empirical NTKs
    ensemble_key = random.split(key, ensemble_size)
    _, train_loss, _, emp_ntk_draw_list, apply_fn_list = vmap(draw_emp_sg_ntk, in_axes=(0,None))(ensemble_key, max(list_training_steps))
    
    return train_loss, emp_ntk_draw_list, apply_fn_list

def calc_plot_sgl_vs_sgntk(test, train, train_indices, training_steps, n, key, net_key, sg_string, kappa, ensemble_size_big, learning_rate=0.1):

    # Define network
    shape = (dim, n, n, 1)
    
    if sg_string == 'erf':
        surr_fun = surr_Sign_Erf
    else:
        surr_fun = surr_Sign_Id
        
    init_fn_surr, apply_fn_surr, kernel_fn_surr = stax.serial(
        stax.Dense(shape[1], W_std=sigma_w, b_std=sigma_b), surr_fun(), #surr_Erf
        stax.Dense(shape[2], W_std=sigma_w, b_std=sigma_b), surr_fun(), #stax.Erf
        stax.Dense(shape[3], W_std=sigma_w*kappa, b_std=sigma_b*kappa)
    )

    apply_fn_surr = jit(apply_fn_surr)
    kernel_fn_surr = jit(kernel_fn_surr, static_argnames='get')

    # Init params (and therefore the network)
    _, params = init_fn_surr(net_key, (-1,dim))

    # Analytic SG-NTK
    k_test_test = erf_ntk(test[0], test[0], len(shape)-1, -1, 1, sg_string)
    k_test_train = k_test_test[:,train_indices]
    k_train_train = k_test_train[train_indices,:]
    
    Z = np.linalg.solve(k_train_train.T, k_test_train.T)
    mu = Z.T @ train[1]

    predict_fn = nt.predict.gradient_descent_mse_ensemble(kernel_fn_surr, train[0] ,train[1])
    mu_nngp, _ = predict_fn(x_test=test[0], get='nngp', compute_cov=True)
    mu_nngp = np.reshape(mu_nngp, (-1,))
    
    nngp_mat = kernel_fn_surr(test[0], test[0], 'nngp')
    cov = nngp_mat - nngp_mat[:,train_indices] @ Z - Z.T @ nngp_mat[train_indices,:] + Z.T @ nngp_mat[train_indices,:][:,train_indices] @ Z
    std = np.sqrt(np.diag(cov)).reshape((-1,1))

    opt_init, opt_update, get_params = optimizers.sgd(learning_rate)
    opt_update = jit(opt_update)

    loss = jit(lambda params, x, y: 0.5 * np.mean((apply_fn_surr(params, x) - y) ** 2))
    grad_loss = jit(lambda state, x, y: grad(loss)(get_params(state), x, y))
    
    # Define function training the network
    def train_sgl(key, training_steps):
        train_losses = []
        test_losses = []
        emp_sg_ntk_draw_list = []

        _, params = init_fn_surr(key, (-1, dim))
        opt_state = opt_init(params)

        for i in range(training_steps+1):
            train_losses += [np.reshape(loss(get_params(opt_state), *train), (1,))]
            test_losses += [np.reshape(loss(get_params(opt_state), *test), (1,))]
            if i < training_steps:
              opt_state = opt_update(i, grad_loss(opt_state, *train), opt_state)

        if training_steps > 0:
            train_losses = jnp.concatenate(train_losses)
            test_losses = jnp.concatenate(test_losses)

        return get_params(opt_state), train_losses, test_losses, apply_fn_surr(get_params(opt_state), test[0])

    ensemble_key = random.split(key, ensemble_size_big)
    _, train_loss, _, apply_fn_list = vmap(train_sgl, in_axes=(0,None))(ensemble_key, training_steps)
    
    return train_loss, apply_fn_list, mu, mu_nngp, std