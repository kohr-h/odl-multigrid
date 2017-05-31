"""Tomography using one phantom at multiple local resolutions."""

import numpy as np
import odl
import odl_multigrid as multigrid

# Define reco space
vol_size = np.array([230.0, 230.0])
vol_min = np.array([-115.0, -115.0])
shape_fbp = (512, 512)
space_fbp = odl.uniform_discr(vol_min, vol_min + vol_size, shape_fbp)

# Set paths and file names
data_path = '/home/hkohr/SciData/Head_CT_Sim/'
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

# Low-resolution space covering the same domain
space_lowres = odl.uniform_discr_fromdiscr(space_fbp, shape=(64, 64))

# Define the high-resolution space for the detail region.
# Find partition indices closest to the desired min and max points, such
# that there is no overlap.
detail_min_pt = [-39, -55]
detail_max_pt = [41, 25]
detail_min_pt_idx = np.floor((detail_min_pt - space_lowres.min_pt) /
                             space_lowres.cell_sides)
detail_min_pt_idx = detail_min_pt_idx.astype(int)
detail_max_pt_idx = np.ceil((detail_max_pt - space_lowres.min_pt) /
                            space_lowres.cell_sides)
detail_max_pt_idx = detail_max_pt_idx.astype(int)
detail_shape_coarse = detail_max_pt_idx - detail_min_pt_idx
detail_min_pt = (space_lowres.min_pt +
                 detail_min_pt_idx * space_lowres.cell_sides)
detail_max_pt = (space_lowres.min_pt +
                 detail_max_pt_idx * space_lowres.cell_sides)
detail_shape = 16 * detail_shape_coarse

space_detail = odl.uniform_discr(detail_min_pt, detail_max_pt, detail_shape)
ray_trafo_detail = odl.tomo.RayTransform(space_detail, geometry,
                                         impl='astra_cuda')

# Masking operator for the detail part in the low-resolution space
mask = multigrid.operators.MaskingOperator(space_lowres,
                                           detail_min_pt, detail_max_pt)
ray_trafo_lowres = odl.tomo.RayTransform(space_lowres, geometry,
                                         impl='astra_cuda')
ray_trafo_lowres_masked = ray_trafo_lowres * mask

# Combine both ray transforms via summation (=reduction)
ray_trafo_combined = odl.ReductionOperator(ray_trafo_lowres_masked,
                                           ray_trafo_detail)
pspace = ray_trafo_combined.domain

# %% Reconstruction
reco_method = 'TV'
timing = True

if reco_method == 'CG':
    callback = (odl.solvers.CallbackPrintIteration(step=2) &
                odl.solvers.CallbackShow(step=2))
    niter_cg = 10
    reco = pspace.zero()
    if timing:
        callback = None
        with odl.util.Timer(reco_method):
            odl.solvers.conjugate_gradient_normal(
                ray_trafo_combined, reco, data, niter=niter_cg,
                callback=callback)
    else:
        odl.solvers.conjugate_gradient_normal(
            ray_trafo_combined, reco, data, niter=niter_cg,
            callback=callback)
    multigrid.graphics.show_both(*reco)

elif reco_method == 'TV':
    grad_detail = odl.Gradient(space_detail, pad_mode='order1')

    l2_norm_sq = odl.solvers.L2NormSquared(ray_trafo_combined.range)
    data_func = l2_norm_sq.translated(data)

    reg_func_lowres = odl.solvers.ZeroFunctional(space_lowres)
    comp_proj_lowres = odl.ComponentProjection(pspace, 0)

    reg_param_detail = 3e-3
    reg_func_detail = (reg_param_detail *
                       odl.solvers.L1Norm(grad_detail.range))
    comp_proj_detail = odl.ComponentProjection(pspace, 1)

    min_val, max_val = np.min(reco_fbp), np.max(reco_fbp)
    box_constr = odl.solvers.IndicatorBox(pspace, min_val, max_val)
    f = odl.solvers.ZeroFunctional(pspace)

    L = [ray_trafo_combined, comp_proj_lowres, grad_detail * comp_proj_detail]
    g = [data_func, reg_func_lowres, reg_func_detail]

    ray_trafo_norm = 1.2 * odl.power_method_opnorm(ray_trafo_combined,
                                                   maxiter=4)
    print('norm of the ray transform: {}'.format(ray_trafo_norm))
    grad_xstart = odl.phantom.shepp_logan(grad_detail.domain, modified=True)
    grad_norm = 1.5 * odl.power_method_opnorm(grad_detail,
                                              xstart=grad_xstart,
                                              maxiter=4)
    print('norm of the gradient: {}'.format(grad_norm))

    # tau and sigma are like step sizes
    tau = 1e-3
    sigma_ray = 1.1 / (tau * ray_trafo_norm ** 2)
    sigma_ident = 0.1 / tau
    sigma_grad = 0.8 / (tau * grad_norm ** 2)
    sigma = [sigma_ray, sigma_ident, sigma_grad]
    lam = 1.5
    # Here we check the convergence criterion for the Douglas-Rachford solver
    check_value = tau * (
        sigma_ray * ray_trafo_norm ** 2 +
        sigma_ident +
        sigma_grad * grad_norm ** 2)
    print('check_value = {}, must be < 4 for convergence'.format(check_value))
    convergence_criterion = check_value < 4
    assert convergence_criterion

    callback = (odl.solvers.CallbackPrintIteration(step=2) &
                odl.solvers.CallbackPrint(g[0] * L[0], fmt='data fit:   {}') &
                odl.solvers.CallbackPrint(g[1] * L[1], fmt='reg lowres: {}') &
                odl.solvers.CallbackPrint(g[2] * L[2], fmt='reg detail: {}') &
                odl.solvers.CallbackShow(step=2, clim=[0.019, 0.023]))

    # Start value, resample & resize FBP reco
    resample_lowres = odl.Resampling(reco_fbp.space, space_lowres)
    reco = pspace.element([resample_lowres(reco_fbp),
                           0.021 * space_detail.one()])

    if timing:
        callback = None
        with odl.util.Timer(reco_method):
            odl.solvers.douglas_rachford_pd(
                reco, f, g, L, tau, sigma, lam=lam, niter=80,
                callback=callback)
    else:
        odl.solvers.douglas_rachford_pd(
            reco, f, g, L, tau, sigma, lam=lam, niter=100,
            callback=callback)

elif reco_method == 'TV_FBP':
    # Compute FBP reco, reproject (masked) and subtract from the data.
    # That should give a good approximation to the data for the insert
    # only.
    # We need to pick a very small frequency window since the discretization
    # is so coarse.
    # TODO: can we compute a reasonable value?
    fbp_op_lowres = odl.tomo.fbp_op(ray_trafo_lowres,
                                    filter_type='Shepp-Logan',
                                    frequency_scaling=0.6)
    fbp_reco_lowres = fbp_op_lowres(data)
    fbp_reco_lowres_reproj = ray_trafo_lowres_masked(fbp_reco_lowres)
    data_detail = data - fbp_reco_lowres_reproj

    # Now we only look at the insert
    grad_detail = odl.Gradient(space_detail, pad_mode='order1')

    l2_norm_sq = odl.solvers.L2NormSquared(data_detail.space)
    data_func = l2_norm_sq.translated(data_detail)

    reg_param_detail = 3e-3
    reg_func_detail = (reg_param_detail *
                       odl.solvers.L1Norm(grad_detail.range))

    f = odl.solvers.ZeroFunctional(space_detail)

    L = [ray_trafo_detail, grad_detail]
    g = [data_func, reg_func_detail]

    ray_trafo_norm = 1.2 * odl.power_method_opnorm(ray_trafo_detail,
                                                   maxiter=4)
    print('norm of the ray transform: {}'.format(ray_trafo_norm))
    grad_xstart = odl.phantom.shepp_logan(grad_detail.domain, modified=True)
    grad_norm = 1.5 * odl.power_method_opnorm(grad_detail,
                                              xstart=grad_xstart,
                                              maxiter=4)

    tau = 1e-3
    sigma_ray = 1.1 / (tau * ray_trafo_norm ** 2)
    sigma_grad = 0.8 / (tau * grad_norm ** 2)
    sigma = [sigma_ray, sigma_grad]
    lam = 1.5
    # Here we check the convergence criterion for the Douglas-Rachford solver
    check_value = tau * (
        sigma_ray * ray_trafo_norm ** 2 +
        sigma_grad * grad_norm ** 2)
    print('check_value = {}, must be < 4 for convergence'.format(check_value))
    convergence_criterion = check_value < 4
    assert convergence_criterion

    callback = (odl.solvers.CallbackPrintIteration(step=2) &
                odl.solvers.CallbackPrint(g[0] * L[0], fmt='data fit:   {}') &
                odl.solvers.CallbackPrint(g[1] * L[1], fmt='reg detail: {}') &
                odl.solvers.CallbackShow(step=2, clim=[0.019, 0.023]))

    # Start value, within plot window
    reco = 0.021 * space_detail.one()

    if timing:
        callback = None
        with odl.util.Timer(reco_method):
            odl.solvers.douglas_rachford_pd(
                reco, f, g, L, tau, sigma, lam=lam, niter=80,
                callback=callback)
    else:
        odl.solvers.douglas_rachford_pd(
            reco, f, g, L, tau, sigma, lam=lam, niter=100,
            callback=callback)
