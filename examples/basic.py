"""Basic multigrid tomography example."""

import numpy as np
import odl
import odl_multigrid as multigrid

# %%
coarse_discr = odl.uniform_discr([-10, -10], [10, 10], [50, 50])
fine_min = [-1, -1]
fine_max = [1, 1]
fine_discr = odl.uniform_discr(fine_min, fine_max, [100, 100])

angle_partition = odl.uniform_partition(0, np.pi, 180)
det_partition = odl.uniform_partition(-15, 15, 3000)

geometry = odl.tomo.Parallel2dGeometry(angle_partition, det_partition,
                                       det_pos_init=[20, 0])

ray_trafo_coarse = odl.tomo.RayTransform(coarse_discr, geometry,
                                         impl='astra_cpu')

coarse_mask = multigrid.operators.MaskingOperator(coarse_discr,
                                                  fine_min, fine_max)
masked_ray_trafo_coarse = ray_trafo_coarse * coarse_mask

ray_trafo_fine = odl.tomo.RayTransform(fine_discr, geometry,
                                       impl='astra_cpu')

pspace_ray_trafo = odl.ReductionOperator(masked_ray_trafo_coarse,
                                         ray_trafo_fine)
pspace = pspace_ray_trafo.domain
phantom_c = odl.phantom.shepp_logan(coarse_discr, modified=True)
phantom_f = odl.phantom.shepp_logan(fine_discr, modified=True)
phantom = pspace.element([phantom_c, phantom_f])

data = pspace_ray_trafo([phantom_c, phantom_f])
data.show('data')

noisy_data = data + odl.phantom.white_noise(ray_trafo_coarse.range, stddev=0.1)
noisy_data.show('noisy data')

reco = pspace_ray_trafo.domain.zero()

# %% CG reconstruction, straight off
callback = (odl.solvers.CallbackPrintIteration(step=2) &
            odl.solvers.CallbackShow(step=2))
odl.solvers.conjugate_gradient_normal(pspace_ray_trafo, reco, data, niter=20,
                                      callback=callback)

# %%
fine_grad = odl.Gradient(fine_discr, pad_mode='order1')

# Differentiable part, build as ||. - g||^2 o P
data_func = odl.solvers.L2NormSquared(
    pspace_ray_trafo.range).translated(noisy_data) * pspace_ray_trafo
reg_param_1 = 7e-3
reg_func_1 = reg_param_1 * (odl.solvers.L2NormSquared(coarse_discr) *
                            odl.ComponentProjection(pspace, 0))
smooth_func = data_func + reg_func_1

# Non-differentiable part composed with linear operators
reg_param = 7e-4
nonsmooth_func = reg_param * odl.solvers.L1Norm(fine_grad.range)

# Assemble into lists (arbitrary number can be given)
comp_proj_1 = odl.ComponentProjection(pspace, 1)
lin_ops = [fine_grad * comp_proj_1]
nonsmooth_funcs = [nonsmooth_func]

# TODO: add to nonsmooth_funcs
box_constr = odl.solvers.IndicatorBox(fine_discr,
                                      np.min(phantom_f), np.max(phantom_f))
f = odl.solvers.ZeroFunctional(pspace)

# %%
# eta^-1 is the Lipschitz constant of the smooth functional gradient
ray_trafo_norm = 1.1 * odl.power_method_opnorm(pspace_ray_trafo,
                                               xstart=phantom, maxiter=2)
print('norm of the ray transform: {}'.format(ray_trafo_norm))
eta = 1 / (2 * ray_trafo_norm ** 2 + 2 * reg_param_1)
print('eta = {}'.format(eta))
grad_norm = 1.1 * odl.power_method_opnorm(fine_grad, xstart=phantom_f,
                                          maxiter=4)
print('norm of the gradient: {}'.format(grad_norm))

# %%
# tau and sigma are like step sizes
sigma = 5e-3
tau = 2 * sigma
# Here we check the convergence criterion for the forward-backward solver
# 1. This is required such that the square root is well-defined
print('tau * sigma * grad_norm ** 2 = {}, should be <= 1'
      ''.format(tau * sigma * grad_norm ** 2))
assert tau * sigma * grad_norm ** 2 <= 1
# 2. This is the actual convergence criterion
check_value = (2 * eta * min(1 / tau, 1 / sigma) *
               np.sqrt(1 - tau * sigma * grad_norm ** 2))
print('check_value = {}, must be > 1 for convergence'.format(check_value))
convergence_criterion = check_value > 1
assert convergence_criterion
# %%
callback = odl.solvers.CallbackShow(step=2)
x = pspace.zero()  # starting point
odl.solvers.forward_backward_pd(x, f=f, g=nonsmooth_funcs, L=lin_ops,
                                h=smooth_func,
                                tau=tau, sigma=[sigma], niter=60,
                                callback=callback)

multigrid.graphics.show_both(*x)
