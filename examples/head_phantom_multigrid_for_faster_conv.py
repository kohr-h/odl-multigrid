"""Tomography using multiple resolutions for the full phantom."""

from collections import namedtuple
from itertools import zip_longest
import numpy as np
import odl

# Define reco space
vol_size = np.array([230.0, 230.0])
vol_min = np.array([-115.0, -115.0])
shape_fbp = (512, 512)
space_fbp = odl.uniform_discr(vol_min, vol_min + vol_size, shape_fbp)

# Set paths and file names
data_path = '/export/scratch2/kohr/data/Head_CT_Sim/'
# data_path = '/home/hkohr/SciData/Head_CT_Sim/'
data_fname = 'HelicalSkullCT_70100644Phantom_no_bed_Dose150mGy_2D_120kV.npy'

# Explicitly instantiate geometry since unpickling is very fragile
angle_partition = odl.uniform_partition(0, 2 * np.pi, 4000)
detector_partition = odl.uniform_partition(-300, 300, 500)
src_radius = 542.8
det_radius = 542.8
geometry = odl.tomo.FanFlatGeometry(angle_partition, detector_partition,
                                    src_radius, det_radius,
                                    src_to_det_init=[-1, 0],
                                    det_axis_init=[0, -1])

# Load data
data_arr = np.load(data_path + data_fname).astype('float32')
log_data_arr = -np.log(data_arr / np.max(data_arr))

# Define ray transform for FBP
ray_trafo_fbp = odl.tomo.RayTransform(space_fbp, geometry)

# Initialize data as ODL space element and display it, clipping to a
# somewhat reasonable range
data = ray_trafo_fbp.range.element(log_data_arr)
data.show('Sinogram', clim=[0, 4.5])

# Compute FBP reco for a good initial guess and for reference
fbp = odl.tomo.fbp_op(ray_trafo_fbp, padding=True, filter_type='Hamming',
                      frequency_scaling=0.8)
reco_fbp = fbp(data)
reco_fbp.show('FBP reconstruction', clim=[0.019, 0.023])
min_val, max_val = np.min(reco_fbp), np.max(reco_fbp)


# %% Reconstruction

timing = False

# Make a sequence of spaces of different resolutions, covering the whole
# volume
space_hires = odl.uniform_discr(vol_min, vol_min + vol_size,
                                shape=(512, 512))
space_midres = odl.uniform_discr(vol_min, vol_min + vol_size,
                                 shape=(128, 128))
space_lowres = odl.uniform_discr(vol_min, vol_min + vol_size,
                                 shape=(32, 32))

# TODO: find reasonable parameters
ResLevel = namedtuple(
    'ResLevel',
    ['space', 'num_iter', 'regularizer', 'reg_param',
     'sigma_ray', 'sigma_grad'])
res_levels = [ResLevel(space_lowres, num_iter=300, regularizer='TV',
                       reg_param=1.2e-5, sigma_ray=0, sigma_grad=0),
              ResLevel(space_midres, num_iter=300, regularizer='TV',
                       reg_param=2e-4, sigma_ray=0, sigma_grad=0),
              ResLevel(space_hires, num_iter=100, regularizer='TV',
                       reg_param=1e-4, sigma_ray=0, sigma_grad=0),
              ]

ray_trafo = odl.tomo.RayTransform(space_hires, geometry, impl='astra_cuda')
ray_trafo_norm = 1.2 * odl.power_method_opnorm(ray_trafo, maxiter=4)
print('norm of the ray transform: {}'.format(ray_trafo_norm))
tau = 1e-2


def check_params(res_level):
    """Check the convergence criterion for the DR solver at ``res_level``."""
    grad = odl.Gradient(res_level.space, pad_mode='order1')
    grad_xstart = odl.phantom.shepp_logan(grad.domain, modified=True)
    grad_norm = 1.5 * odl.power_method_opnorm(grad, xstart=grad_xstart,
                                              maxiter=10)
    print('norm of the gradient: {}'.format(grad_norm))

    res_level = ResLevel(res_level.space, res_level.num_iter,
                         res_level.regularizer, res_level.reg_param,
                         sigma_ray=1.5 / tau,
                         sigma_grad=1.5 / (tau * grad_norm ** 2))

    # Here we check the convergence criterion for the Douglas-Rachford solver
    check_value = tau * (res_level.sigma_ray +
                         res_level.sigma_grad * grad_norm ** 2)
    print('check_value = {}, must be < 4 for convergence'.format(check_value))
    convergence_criterion = check_value < 4
    assert convergence_criterion

    return res_level

for i, res_level in enumerate(res_levels):
    res_levels[i] = check_params(res_level)

lam = 1.0

# Rescale data
data /= ray_trafo_norm

# Start value at the very beginning. We choose a constant inside the display
# window so we see what's happening from the beginning.
resampling = odl.Resampling(reco_fbp.space, res_levels[0].space)
reco = resampling(reco_fbp)

for cur_res, next_res in zip_longest(res_levels, res_levels[1:]):
    # Functionals composed with operators, given in split form
    ray_trafo = odl.tomo.RayTransform(cur_res.space, geometry,
                                      impl='astra_cuda')
    ray_trafo = ray_trafo / ray_trafo_norm
    l2_norm_sq = odl.solvers.L2NormSquared(ray_trafo.range)
    # TODO: bin data?
    data_func = l2_norm_sq.translated(data)

    if cur_res.regularizer == 'L2':
        reg_func = cur_res.reg_param * odl.solvers.L2NormSquared(cur_res.space)
        reg_op = odl.IdentityOperator(cur_res.space)
    elif cur_res.regularizer == 'H1':
        grad = odl.Gradient(cur_res.space, pad_mode='order1')
        l2_norm_sq = odl.solvers.L2NormSquared(grad.range)
        reg_func = cur_res.reg_param * l2_norm_sq
        reg_op = grad
    elif cur_res.regularizer == 'TV':
        grad = odl.Gradient(cur_res.space, pad_mode='order1')
        l1_norm = odl.solvers.L1Norm(grad.range)
        reg_func = cur_res.reg_param * l1_norm
        reg_op = grad
        print('reg_param:', cur_res.reg_param)
    else:
        assert False

    # Assemble into lists for solver (problem has to be specified in
    # split form)
    L = [ray_trafo, reg_op]
    g = [data_func, reg_func]
    sigma = [cur_res.sigma_ray, cur_res.sigma_grad]

    box_constr = odl.solvers.IndicatorBox(cur_res.space, min_val, max_val)
    f = box_constr

    # Show stuff during iteration
    callback = (odl.solvers.CallbackPrintIteration(step=2) &
                odl.solvers.CallbackPrint(data_func * ray_trafo) &
                odl.solvers.CallbackPrint(reg_func * reg_op) &
                odl.solvers.CallbackShow(step=2, clim=[0.019, 0.023]))

    if timing:
        callback = None
        with odl.util.Timer():
            odl.solvers.douglas_rachford_pd(reco, f, g, L, tau, sigma, lam=lam,
                                            niter=cur_res.num_iter,
                                            callback=callback)
    else:
        odl.solvers.douglas_rachford_pd(reco, f, g, L, tau, sigma, lam=lam,
                                        niter=cur_res.num_iter,
                                        callback=callback)

    # Resample reco to the space of the next iteration
    if next_res is not None:
        resampling = odl.Resampling(cur_res.space, next_res.space)
        reco = resampling(reco)
        # reco *= cur_res.space.cell_volume / next_res.space.cell_volume
