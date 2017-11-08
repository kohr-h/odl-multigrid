"""Multi-resolution tomography with "russian doll" Shepp-Logan phantom.

In this variant, we use two datasets, one low-resolution "overview" scan
and one high-resolution "ROI" scan. The task is to reconstruct the
detail region without the missing data problem of region-of-interest
tomography.
We consider two methods to solve this problem:

1. Subtract the contribution of the surrounding volume from the data
   corresponding to the detail region.
2. Reconstruct jointly the overview and detail regions by using the
   information about the surrounding volume in the forward projection
   of the detail region.
"""


import numpy as np
import odl
import odl_multigrid as multigrid


# %% Notation

# --- Problem formulation --- #

# Spaces
# X1 = low-resolution reconstruction space, full volume
# Y1 = projection space, zoomed-out scan
# X2 = high-resolution reconstruction space, ROI
# Y2 = projection space, zoomed-in scan

# Operators
# R11 = ray trafo X1 -> Y1 (low-res, zoomed out)
# R22 = ray trafo X2 -> Y2 (hi-res, zoomed in, ROI only)

# Projection data
# g1 = projection data, zoomed out
# g2 = projection data, zoomed in, with outer contribution

# --- Data generation --- #

# Space
# X1_hires = high-resolution reconstruction space, full volume

# Phantom
# f1_hires = high-resolution phantom, full volume

# Operators
# R11_hires = ray trafo X1_hires -> Y1 (hi-res, zoomed out, to generate g1)
# R21_hires = ray trafo X1_hires -> X2 (hi-res, zoomed in, to generate g2)

# --- Other --- #

# Phantoms (for comparison)
# f1 = low-resolution phantom, full volume
# f2 = high-resolution ROI part of the phantom

# Projection data
# g1_lowres = projection data, zoomed out, from low-res input
# g2_no_outer = projection data, zoomed in, no outer contribution

# %% Quantities that are independent of the scan

# Detector of length 20 [cm] and 1024 pixels
det_part = odl.uniform_partition(-10, 10, 1024)

# Ray transform implementation (`None` means "take fastest available")
impl = None

# %% Define X1, Y1 and R11

# The size of the spatial region is such that it would fit onto the detector
# in parallel beam geometry (14 x 14 [cm^2]).
X1 = odl.uniform_discr([-7, -7], [7, 7], shape=(128, 128))
print('X1_px_size / Y1_px_size:', X1.cell_sides[0] / det_part.cell_sides[0])

# Full scan, 1 degree increment
# TODO: this can probably be made significantly coarser
Y1_angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)
Y1_angle_part = odl.nonuniform_partition(Y1_angles, min_pt=0, max_pt=2 * np.pi)
# Put the detector close to the object to keep the magnification low.
# Here it is roughly 1.2 ((50 + 10) / 50).
Y1_geom = odl.tomo.FanFlatGeometry(Y1_angle_part, det_part,
                                   src_radius=50, det_radius=10)
print('Y1 magnification:',
      (Y1_geom.src_radius + Y1_geom.det_radius) / Y1_geom.src_radius)
R11 = odl.tomo.RayTransform(X1, Y1_geom, impl=impl)

# Show footprint of a 10 x 10 [cm^2] square, this should not go outside
# the detector region.
R11(odl.phantom.cuboid(X1, [-5, -5], [5, 5])).show(
    'Detector footprint of a 10x10 square')

# %% Define X2, Y2 and R22

# Fine discretization for the ROI scan. We put the ROI into the upper
# right corner, with size 1.4 x 1.4 [cm^2]. The resolution is about 4.2 times
# the detector resolution, matching the 4x magnification (see below).
X2_min_pt = [-2.1875, -0.875]
X2_max_pt = [-0.7875, 0.525]
X2 = odl.uniform_discr(X2_min_pt, X2_max_pt, shape=(256, 256))
print('X2_px_size / Y2_px_size:', X2.cell_sides[0] / det_part.cell_sides[0])

# Full scan, 1 degree increment
Y2_angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)
Y2_angle_part = odl.nonuniform_partition(Y2_angles, min_pt=0, max_pt=2 * np.pi)

# Put the source closer and move the detector out. This way we achieve
# 4x magnification ((25 + 75) / 25).
# TODO: make the detector move according to the projection of the ROI
Y2_geom = odl.tomo.FanFlatGeometry(Y2_angle_part, det_part,
                                   src_radius=25, det_radius=75)
print('Y2 magnification:',
      (Y2_geom.src_radius + Y2_geom.det_radius) / Y2_geom.src_radius)
R22 = odl.tomo.RayTransform(X2, Y2_geom, impl=impl)

# Show footprint of the region of interest, should also fit on the detector
R22(X2.one()).show('ROI footprint')

# %% Make phantoms

# Merged low-resolution phantom
f1 = odl.phantom.shepp_logan(X1, modified=True)
f1 += odl.phantom.shepp_logan(X1, modified=True,
                              min_pt=X2.min_pt, max_pt=X2.max_pt)

f1.show('f1 -- low-res full phantom')
f1.show('low-res ROI part in f1', coords=[X2.min_pt, X2.max_pt])

# High-resolution full phantom (for data generation)
X1_hires = odl.uniform_discr_fromdiscr(X1, cell_sides=X2.cell_sides)
f1_hires = odl.phantom.shepp_logan(X1_hires, modified=True)
f1_hires += odl.phantom.shepp_logan(
    X1_hires, modified=True, min_pt=X2.min_pt, max_pt=X2.max_pt)

f1_hires.show('f1_hires -- hi-res full phantom')

# ROI high-resolution phantom
f2 = odl.phantom.shepp_logan(X2, modified=True)
f2.show('f2 -- hi-res ROI part')

# %% Generate projection data

# Full data from low-resolution phantom
g1_lowres = R11(f1)
g1_lowres.show('g1_lowres -- zoomed-out data from low-res phantom')

# Full data from high-resolution phantom
R11_hires = odl.tomo.RayTransform(X1_hires, Y1_geom, impl=impl)
g1 = R11_hires(f1_hires)
g1.show('g1 -- zoomed-out data from hi-res phantom')

# Fictive ROI-only projection data
g2_no_outer = R22(f2)
g2_no_outer.show('g2_no_outer -- fictive zoomed-in ROI-only data')

# Actual ROI projection data (including outer values)
R21_hires = odl.tomo.RayTransform(X1_hires, Y2_geom, impl=impl)
g2 = R21_hires(f1_hires)
g2.show('g2 -- zoomed-in data from hi-res phantom')

# %% Make FBP reco and reproject masked version for modified ROI data

# FBP from ROI projection data only (no data modification)
FBP2 = odl.tomo.fbp_op(R22, filter_type='Hann', frequency_scaling=0.8)
f2_FBP2 = FBP2(g2)
f2_FBP2.show('f2_FBP2 -- ROI FBP reconstruction', clim=[0, 1])

# Compute low-res FBP reco of the full volume
FBP1 = odl.tomo.fbp_op(R11, filter_type='Hann', frequency_scaling=0.5)
f1_FBP1 = FBP1(g1)
f1_FBP1.show('f1_FBP1 -- low-res full FBP reconstruction')

# Forward project the masked reco to Y2
M21 = multigrid.operators.MaskingOperator(X1, X2.min_pt, X2.max_pt)
R21 = odl.tomo.RayTransform(X1, Y2_geom, impl=impl)
A21 = R21 * M21
g2_outer_FBP1 = A21(f1_FBP1)
g2_outer_FBP1.show('g2_outer_FBP1 -- Reprojected masked full FBP reco')

# Modify ROI data by subtracting reprojected outer reco
g2_no_outer_FBP1 = g2 - g2_outer_FBP1
g2_no_outer_FBP1.show('g2_no_outer_FBP1 -- Modified ROI data')

# Mask the part where there should not be any data
g2_mask = R22.range.element(np.greater(R22(X2.one()), 0))
g2_no_outer_FBP1_masked = g2_no_outer_FBP1 * g2_mask
g2_no_outer_FBP1_masked[g2_no_outer_FBP1_masked.asarray() < 0] = 0
g2_no_outer_FBP1_masked.show('g2_no_outer_FBP1_masked')
f2_FBP2_no_outer_masked = FBP2(g2_no_outer_FBP1_masked)
f2_FBP2_no_outer_masked.show('f2_FBP2_no_outer_masked')

# %% Make FBP reco of the ROI with modified data

# Make FBP reco from modified data
f2_FBP2_no_outer = FBP2(g2_no_outer_FBP1)
f2_FBP2_no_outer.show('f2_FBP2_no_outer -- ROI FBP reco from modified data')
f2_FBP2_no_outer.show('f2_FBP2_no_outer profile y=0', coords=[None, 0])

# Show multi-resolution phantom
multigrid.graphics.show_both(f1_FBP1, f2_FBP2_no_outer)

# %% Make TV reco of the ROI with original data
#
# Solving the ROI problem
#
#     f2_tv_orig = argmin{f} [ D( R22(f) ) + alpha * TV(f) + i_nonneg(f) ]
#
# with D(g) = ||g - g2||_2^2 and i_nonneg = indicator function enforcing
# nonnegativity of f.

# Functionals
# D = data matching functional: Y2 -> R, ||. - g2||_2^2
# S = (alpha * L12-Norm): X2^2 -> R, for isotropic TV
# P = i_nonneg: X2 -> R

# Operators
# G = spatial gradient X2 -> X2^2

G = odl.Gradient(X2, pad_mode='symmetric')

D = odl.solvers.L2NormSquared(R22.range).translated(g2)
alpha = 1e-3
S = alpha * odl.solvers.GroupL1Norm(G.range)
P = odl.solvers.IndicatorBox(X2, 0, np.inf)

# Arguments for the solver
f_func = P
g_funcs = [D, S]
L_ops = [R22, G]

# Operator norm estimation for the step size parameters
R22_norm = odl.power_method_opnorm(R22, maxiter=10)
# TODO: choose a different starting point
G_norm = odl.power_method_opnorm(G, xstart=f2_FBP2, maxiter=10)

# We need tau * sum[i](sigma_i * opnorm_i^2) < 4 for convergence, so we
# choose tau and set sigma_i = c / (tau * opnorm_i^2) such that sum[i](c) < 4
tau = 1.0
opnorms = [R22_norm, G_norm]
sigmas = [3.0 / (tau * len(opnorms) * opnorm ** 2) for opnorm in opnorms]

callback = (odl.solvers.CallbackPrintIteration(step=20) &
            odl.solvers.CallbackShow(step=20, clim=[0, 3]))

f_tv_orig = f2_FBP2.copy()
odl.solvers.douglas_rachford_pd(
    f_tv_orig, f_func, g_funcs, L_ops, tau, sigmas, niter=300,
    callback=callback)

f_tv_orig.show('f_tv_orig -- ROI TV reco from ROI data only', clim=[0, 3])
fig = f2.show('f2 profile at y=0', coords=[None, 0])
f_tv_orig.show(coords=[None, 0], fig=fig)


# %% Make TV reco of the ROI with modified data
#
# Solving the problem
#
#     f2_tv_mod = argmin_f [ D( R22(f) ) + alpha * TV(f) + i_nonneg(f) ]
#
# with D(g) = ||g - g2_mod||_2^2 and i_nonneg = indicator function enforcing
# nonnegativity of f, and g2_mod = ROI data modified by re-projection.

# Functionals
# D = data matching functional: Y2 -> R, ||. - g2_mod||_2^2
# S = (alpha * L12-Norm): X2^2 -> R, for isotropic TV
# P = i_nonneg: X2 -> R

# Operators
# G = spatial gradient X2 -> X2^2

G = odl.Gradient(X2, pad_mode='symmetric')

D = odl.solvers.L2NormSquared(R22.range).translated(g2_no_outer_FBP1)
alpha = 3e-4
S = alpha * odl.solvers.GroupL1Norm(G.range)
P = odl.solvers.IndicatorBox(X2, 0, 1)

f_func = P
g_funcs = [D, S]
L_ops = [R22, G]

# Operator norm estimation for the step size parameters
R22_norm = odl.power_method_opnorm(R22, maxiter=10)
# TODO: choose a different starting point
case2_grad_norm = odl.power_method_opnorm(G, xstart=f2_FBP2, maxiter=10)

# We need tau * sum[i](sigma_i * opnorm_i^2) < 4 for convergence
tau = 1.0
opnorms = [R22_norm, case2_grad_norm]
sigmas = [3.0 / (tau * len(opnorms) * opnorm ** 2) for opnorm in opnorms]

callback = (odl.solvers.CallbackPrintIteration(step=20) &
            odl.solvers.CallbackShow(step=20, clim=[0, 1]))

f_tv_mod = f2_FBP2.copy()
odl.solvers.douglas_rachford_pd(
    f_tv_mod, f_func, g_funcs, L_ops, tau, sigmas, niter=200,
    callback=callback)

f_tv_mod.show('f_tv_mod -- ROI TV reco from modified data', clim=[0, 1])
fig = f2.show('f2 profile at y=0', coords=[None, 0])
f_tv_mod.show(coords=[None, 0], fig=fig)


# %% Joint reconstruction with triangular system

# Solving the joint problem
#
#     (f1_tri, f2_tri) = argmin{f1,f2} [ D( A(f1, f2) ) +
#                                        alpha1 * ||grad f_1||_2^2 +
#                                        alpha2 * TV(f2) ]
#
# with D: Y1 x Y2 -> R squared L2-norm, and A: X1 x X2 -> Y1 x Y2,
#
#                 ( A11  0   )   ( f1 )   ( A11(f1)           )
#     A(f1, f2) = (          ) * (    ) = (                   )
#                 ( A21  A22 )   ( f2 )   ( A21(f1) + A22(f2) )
#
# A11 = R11, A22 = R22, A21 = R21 o M21, where M21 is a masking operator
# for the ROI in X1.

# Functionals
# D = data matching functional: Y1 x Y2 -> R, ||. - g1||_Y1^2 + ||. - g2||_Y2^2
# S1 = squared L2-norm: X1^2 -> R, for Tikhonov functional
# S2 = (alpha * L12-Norm): X2^2 -> R, for isotropic TV

# Operators
# A = forward operator "matrix": X1 x X2 -> Y1 x Y2
# G1 = spatial gradient: X1 -> X1^2
# G2 = spatial gradient: X2 -> X2^2
# B1 = G1 extended to X1 x X2, B1(f1, f2) = G1(f1)
# B2 = G2 extended to X1 x X2, B2(f1, f2) = G2(f2)

A = odl.ProductSpaceOperator([[R11, 0],
                              [A21, R22]])
G1 = odl.Gradient(X1, pad_mode='symmetric')
G2 = odl.Gradient(X2, pad_mode='order1')
# Extend gradients to product space
B1 = G1 * odl.ComponentProjection(A.domain, 0)
B2 = G2 * odl.ComponentProjection(A.domain, 1)

# TODO: weighting for differences in sizes (avoid large region domination)
D = odl.solvers.L2NormSquared(A.range).translated([g1, g2])

# For isotropic TV
# alpha1 = 1e-2
# S1 = alpha1 * odl.solvers.GroupL1Norm(G1.range)
# For Tikhonov
alpha1 = 1e-2
S1 = alpha1 * odl.solvers.L2NormSquared(G1.range)
# TV on second component
alpha2 = 1e-4
S2 = alpha2 * odl.solvers.GroupL1Norm(G2.range)

# Arguments for the solver
f_func = odl.solvers.ZeroFunctional(A.domain)  # unused
g_funcs = [D, S1, S2]
L_ops = [A, B1, B2]

# Operator norm estimation for the step size parameters
xstart = A.domain.element([f1_FBP1, f2_FBP2])
A_norm = odl.power_method_opnorm(A, maxiter=10)
B1_norm = odl.power_method_opnorm(B1, xstart=xstart, maxiter=10)
B2_norm = odl.power_method_opnorm(B2, xstart=xstart, maxiter=10)

# We need tau * sum[i](sigma_i * opnorm_i^2) < 4 for convergence, so we
# choose tau and set sigma_i = c / (tau * opnorm_i^2) such that sum[i](c) < 4
tau = 1.0
opnorms = [A_norm, B1_norm, B2_norm]
sigmas = [3.0 / (tau * len(opnorms) * opnorm ** 2) for opnorm in opnorms]

callback = (odl.solvers.CallbackPrintIteration(step=20) &
            odl.solvers.CallbackShow(step=20))

f_tri = A.domain.element([f1_FBP1, f2_FBP2])
odl.solvers.douglas_rachford_pd(
    f_tri, f_func, g_funcs, L_ops, tau, sigmas, niter=600, callback=callback)

f1_tri, f2_tri = f_tri
f1_tri.show('f1_tri -- first component of joint reco - low-res full volume',
            clim=[0, 1])
f2_tri.show('f2_tri -- second component of joint reco - hi-res ROI',
            clim=[0, 1])
fig = f1.show('f1 profile at y=0', coords=[None, 0], label='Phantom')
f1_tri.show(coords=[None, 0], fig=fig, label='Joint reco, triagonal')
fig = f2.show('f2 profile at y=0', coords=[None, 0], label='Phantom')
f2_tri.show(coords=[None, 0], fig=fig, label='Joint reco, triagonal')
