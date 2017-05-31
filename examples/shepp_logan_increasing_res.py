"""Tomography using multiple resolutions for the full phantom."""

from collections import namedtuple
from itertools import zip_longest
import numpy as np
import odl

# Define reco space & phantom
vol_min = np.array([-1.0, -1.0])
vol_max = np.array([1.0, 1.0])
vol_shape_fbp = (512, 512)
space_fbp = odl.uniform_discr(vol_min, vol_max, vol_shape_fbp)

phantom = odl.phantom.shepp_logan(space_fbp, modified=True)

# Geometry
angle_partition = odl.uniform_partition(0, 2 * np.pi, 60)
detector_partition = odl.uniform_partition(-3, 3, 500)
src_radius = 8
det_radius = 8
geometry = odl.tomo.FanFlatGeometry(angle_partition, detector_partition,
                                    src_radius, det_radius)

# Generate data
ray_trafo_fbp = odl.tomo.RayTransform(space_fbp, geometry)
data = ray_trafo_fbp(phantom)
data_noisy = (data +
              0.2 * np.max(data) * odl.phantom.white_noise(data.space))

data_noisy.show('Sinogram')

# Compute FBP reco for a good initial guess and for reference
fbp_op = odl.tomo.fbp_op(ray_trafo_fbp, padding=True, filter_type='Hann',
                         frequency_scaling=0.2)
reco_fbp = fbp_op(data_noisy)
reco_fbp.show('FBP reconstruction')


# %% Reconstruction
timing = False

# Make a sequence of spaces of different resolutions, covering the whole
# volume
space_hires = odl.uniform_discr(vol_min, vol_max, shape=(512, 512),
                                interp='linear')
space_midres = odl.uniform_discr(vol_min, vol_max, shape=(128, 128),
                                 interp='linear')
space_lowres = odl.uniform_discr(vol_min, vol_max, shape=(32, 32),
                                 interp='linear')

# Set sigma parameters initially according to the high-resolution operator
# norms
ray_trafo = odl.tomo.RayTransform(space_hires, geometry,
                                  impl='astra_cuda')
ray_trafo_norm = 1.2 * odl.power_method_opnorm(ray_trafo, maxiter=4)
print('norm of the ray transform: {}'.format(ray_trafo_norm))
grad = odl.Gradient(space_hires, pad_mode='order1')
grad_xstart = odl.phantom.shepp_logan(grad.domain, modified=True)
grad_norm = 1.5 * odl.power_method_opnorm(grad, xstart=grad_xstart,
                                          maxiter=10)
print('norm of the gradient: {}'.format(grad_norm))

ResLevel = namedtuple(
    'ResLevel',
    ['space', 'num_iter', 'regularizer', 'reg_param',
     'sigma_ray', 'sigma_grad'])

tau = 1e-1
lam = 1.5

res_levels = [ResLevel(space_lowres, num_iter=150, regularizer='TV',
                       reg_param=1e-3,
                       sigma_ray=1.5 / (tau * ray_trafo_norm ** 2),
                       sigma_grad=1.5 / (tau * grad_norm ** 2)),
              ResLevel(space_midres, num_iter=200, regularizer='TV',
                       reg_param=2e-4,
                       sigma_ray=0.5 / (tau * ray_trafo_norm ** 2),
                       sigma_grad=2.5 / (tau * grad_norm ** 2)),
              ResLevel(space_hires, num_iter=250, regularizer='TV',
                       reg_param=1e-4,
                       sigma_ray=0.01 / (tau * ray_trafo_norm ** 2),
                       sigma_grad=3.5 / (tau * grad_norm ** 2)),
              ]


def check_params(res_level):
    """Check the convergence criterion for the DR solver at ``res_level``."""
    ray_trafo = odl.tomo.RayTransform(res_level.space, geometry,
                                      impl='astra_cuda')
    ray_trafo_norm = 1.2 * odl.power_method_opnorm(ray_trafo, maxiter=4)
    print('norm of the ray transform: {}'.format(ray_trafo_norm))
    grad = odl.Gradient(res_level.space, pad_mode='order1')
    grad_xstart = odl.phantom.shepp_logan(grad.domain, modified=True)
    grad_norm = 1.5 * odl.power_method_opnorm(grad, xstart=grad_xstart,
                                              maxiter=10)
    print('norm of the gradient: {}'.format(grad_norm))

    # Here we check the convergence criterion for the Douglas-Rachford solver
    check_value = tau * (res_level.sigma_ray * ray_trafo_norm ** 2 +
                         res_level.sigma_grad * grad_norm ** 2)
    print('check_value = {}, must be < 4 for convergence'.format(check_value))
    convergence_criterion = check_value < 4
    assert convergence_criterion


for res_level in res_levels:
    check_params(res_level)

# Start value at the very beginning, the FBP reco resampled to the first space
resampling = odl.Resampling(reco_fbp.space, res_levels[0].space)
reco = resampling(reco_fbp)
# reco = res_levels[0].space.zero()

for cur_res, next_res in zip_longest(res_levels, res_levels[1:]):
    # Functionals composed with operators, given in split form
    ray_trafo = odl.tomo.RayTransform(cur_res.space, geometry,
                                      impl='astra_cuda')
    l2_norm_sq = odl.solvers.L2NormSquared(ray_trafo.range)
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

    box_constr = odl.solvers.IndicatorBox(cur_res.space, 0, 1)
    f = box_constr

    # Show stuff during iteration
    callback = (odl.solvers.CallbackPrintIteration(step=2) &
                odl.solvers.CallbackPrint(data_func * ray_trafo) &
                odl.solvers.CallbackPrint(reg_func * reg_op) &
                odl.solvers.CallbackShow(step=2, clim=[0, 1]))

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
