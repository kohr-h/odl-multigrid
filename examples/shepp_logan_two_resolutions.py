"""Tomography using one phantom at multiple local resolutions."""

import numpy as np
import odl
import odl_multigrid as multigrid

# %%

# Basic discretizations
min_pt = [-10, -10]
max_pt = [10, 10]
coarse_discr = odl.uniform_discr(min_pt, max_pt, [20, 20])
fine_discr = odl.uniform_discr(min_pt, max_pt, [1000, 1000])

insert_min_pt = [-6, -8]
insert_max_pt = [2, -4]

# Geometry
angle_partition = odl.uniform_partition(0, 2 * np.pi, 180)

# Make detector large enough to cover the object
src_radius = 50
det_radius = 50
opening_angle = np.arctan(max(np.max(np.abs(min_pt)), np.max(np.abs(max_pt))) /
                          src_radius)
det_size = np.floor(2 * (src_radius + det_radius) * np.sin(opening_angle))
det_shape = int(det_size / np.min(fine_discr.cell_sides))
det_max_pt = det_size / 2
det_min_pt = -det_max_pt
det_partition = odl.uniform_partition(det_min_pt, det_max_pt, det_shape)
geometry = odl.tomo.FanFlatGeometry(angle_partition, det_partition,
                                    src_radius, det_radius)

# Mask
coarse_mask = multigrid.operators.MaskingOperator(coarse_discr,
                                                  insert_min_pt, insert_max_pt)
coarse_ray_trafo = odl.tomo.RayTransform(coarse_discr, geometry,
                                         impl='astra_cuda')
masked_coarse_ray_trafo = coarse_ray_trafo * coarse_mask

# Phantom
phantom_c = odl.phantom.shepp_logan(coarse_discr, modified=True)
phantom_f = odl.phantom.shepp_logan(fine_discr, modified=True)


# Define insert discretization using the fine cell sizes but the insert
# min and max points
insert_discr = odl.uniform_discr_fromdiscr(
    fine_discr, min_pt=insert_min_pt, max_pt=insert_max_pt,
    cell_sides=fine_discr.cell_sides)

# Restrict the phantom to the insert discr
resizing_operator = odl.ResizingOperator(fine_discr, insert_discr)
phantom_insert = resizing_operator(phantom_f)

# Ray trafo on the insert discretization only
insert_ray_trafo = odl.tomo.RayTransform(insert_discr, geometry,
                                         impl='astra_cuda')

# Forward operator = sum of masked coarse ray trafo and insert ray trafo
sum_ray_trafo = odl.ReductionOperator(masked_coarse_ray_trafo,
                                      insert_ray_trafo)

# Make phantom in the product space
pspace = sum_ray_trafo.domain
phantom = pspace.element([phantom_c, phantom_insert])
multigrid.graphics.show_both(*phantom)

# Create noise-free data
fine_ray_trafo = odl.tomo.RayTransform(fine_discr, geometry,
                                       impl='astra_cuda')
data = fine_ray_trafo(phantom_f)
data.show('data')

# Make noisy data
noisy_data = data + odl.phantom.white_noise(fine_ray_trafo.range, stddev=0.1)
noisy_data.show('noisy data')

# %% Reconstruction
reco_method = 'TV_FBP'
timing = True

if reco_method == 'CG':
    callback = (odl.solvers.CallbackPrintIteration(step=2) &
                odl.solvers.CallbackShow(step=2))
    reco = pspace.zero()
    if timing:
        callback = None
        with odl.util.Timer(reco_method):
            odl.solvers.conjugate_gradient_normal(
                sum_ray_trafo, reco, noisy_data, niter=6, callback=callback)
    else:
        odl.solvers.conjugate_gradient_normal(
            sum_ray_trafo, reco, noisy_data, niter=6, callback=callback)
    multigrid.graphics.show_both(*reco)

elif reco_method == 'TV':
    insert_grad = odl.Gradient(insert_discr, pad_mode='order1')

    # Differentiable part, build as ||. - g||^2 o P
    data_func = odl.solvers.L2NormSquared(
        sum_ray_trafo.range).translated(noisy_data) * sum_ray_trafo
    reg_param_1 = 1e0
    reg_func_1 = reg_param_1 * (odl.solvers.L2NormSquared(coarse_discr) *
                                odl.ComponentProjection(pspace, 0))
    smooth_func = data_func + reg_func_1

    # Non-differentiable part composed with linear operators
    reg_param = 8e-3
    nonsmooth_func = reg_param * odl.solvers.L1Norm(insert_grad.range)

    # Assemble into lists (arbitrary number can be given)
    comp_proj_1 = odl.ComponentProjection(pspace, 1)
    lin_ops = [insert_grad * comp_proj_1]
    nonsmooth_funcs = [nonsmooth_func]

    box_constr = odl.solvers.IndicatorBox(pspace,
                                          np.min(phantom_f),
                                          np.max(phantom_f))
    f = box_constr

    # eta^-1 is the Lipschitz constant of the smooth functional gradient
    ray_trafo_norm = 1.1 * odl.power_method_opnorm(sum_ray_trafo,
                                                   xstart=phantom, maxiter=2)
    print('norm of the ray transform: {}'.format(ray_trafo_norm))
    eta = 1 / (2 * ray_trafo_norm ** 2 + 2 * reg_param_1)
    print('eta = {}'.format(eta))
    grad_norm = 1.1 * odl.power_method_opnorm(insert_grad,
                                              xstart=phantom_insert,
                                              maxiter=4)
    print('norm of the gradient: {}'.format(grad_norm))

    # tau and sigma are like step sizes
    sigma = 4e-3
    tau = 1.0 * sigma
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

    callback = (odl.solvers.CallbackPrintIteration(step=2) &
                odl.solvers.CallbackShow(step=2,
                                         clim=[np.min(phantom_f),
                                               np.max(phantom_f)]))
    reco = pspace.zero()  # starting point
    if timing:
        callback = None
        with odl.util.Timer(reco_method):
            odl.solvers.forward_backward_pd(
                reco, f=f, g=nonsmooth_funcs, L=lin_ops, h=smooth_func,
                tau=tau, sigma=[sigma], niter=150, callback=callback)
    else:
        odl.solvers.forward_backward_pd(
            reco, f=f, g=nonsmooth_funcs, L=lin_ops, h=smooth_func,
            tau=tau, sigma=[sigma], niter=150, callback=callback)

    multigrid.graphics.show_both(reco[0], reco[1])

elif reco_method == 'TV_FBP':
    # Compute FBP reco, reproject (masked) and subtract from the data.
    # That should give a good approximation to the data for the insert
    # only.
    # We need to pick a very small frequency window since the discretization
    # is so coarse.
    # TODO: can we compute a reasonable value?
    coarse_fbp_op = odl.tomo.fbp_op(coarse_ray_trafo,
                                    filter_type='Shepp-Logan',
                                    frequency_scaling=0.02)
    fbp_reco_c = coarse_fbp_op(noisy_data)
    fbp_reco_c_reproj = masked_coarse_ray_trafo(fbp_reco_c)
    insert_only_data = noisy_data - fbp_reco_c_reproj

    # Now we only look at the insert
    insert_grad = odl.Gradient(insert_discr, pad_mode='order1')

    # Differentiable part, build as ||. - g||^2 o P
    l2_norm_sq = odl.solvers.L2NormSquared(insert_ray_trafo.range)
    data_func = l2_norm_sq.translated(insert_only_data) * insert_ray_trafo
    smooth_func = data_func

    # Non-differentiable part composed with linear operators
    reg_param = 8e-3
    nonsmooth_func = reg_param * odl.solvers.L1Norm(insert_grad.range)

    # Assemble into lists (arbitrary number can be given)
    lin_ops = [insert_grad]
    nonsmooth_funcs = [nonsmooth_func]

    box_constr = odl.solvers.IndicatorBox(insert_discr,
                                          np.min(phantom_f),
                                          np.max(phantom_f))
    f = box_constr

    # eta^-1 is the Lipschitz constant of the smooth functional gradient
    ray_trafo_norm = 1.1 * odl.power_method_opnorm(insert_ray_trafo,
                                                   maxiter=10)
    print('norm of the ray transform: {}'.format(ray_trafo_norm))
    eta = 1 / (2 * ray_trafo_norm ** 2 + 2 * reg_param_1)
    print('eta = {}'.format(eta))
    grad_norm = 1.1 * odl.power_method_opnorm(insert_grad,
                                              xstart=phantom_insert,
                                              maxiter=10)
    print('norm of the gradient: {}'.format(grad_norm))

    # tau and sigma are like step sizes
    sigma = 4e-3
    tau = 1.5 * sigma
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

    callback = (odl.solvers.CallbackPrintIteration(step=2) &
                odl.solvers.CallbackShow(step=2,
                                         clim=[np.min(phantom_f),
                                               np.max(phantom_f)]))
    reco = insert_discr.zero()  # starting point
    if timing:
        callback = None
        with odl.util.Timer(reco_method):
            odl.solvers.forward_backward_pd(
                reco, f=f, g=nonsmooth_funcs, L=lin_ops, h=smooth_func,
                tau=tau, sigma=[sigma], niter=150, callback=callback)
    else:
        odl.solvers.forward_backward_pd(
            reco, f=f, g=nonsmooth_funcs, L=lin_ops, h=smooth_func,
            tau=tau, sigma=[sigma], niter=150, callback=callback)

    multigrid.graphics.show_both(fbp_reco_c, reco)
