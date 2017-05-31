"""Find the footprint of a volume on the detector numerically."""

import numpy as np
import odl
import odl_multigrid as multigrid

# %% Set up the operators and phantoms

coarse_discr = odl.uniform_discr([-10, -10], [10, 10], [50, 50],
                                 dtype='float32')
fine_min = [1, 1]
fine_max = [2, 2]
fine_discr = odl.uniform_discr(fine_min, fine_max, [100, 100], dtype='float32')

angle_partition = odl.uniform_partition(0, np.pi, 180, nodes_on_bdry=True)
det_partition = odl.uniform_partition(-20, 20, 4000)

r_src = 500
r_det = 100
d_min = det_partition.min_pt
d_max = det_partition.max_pt
angle_top = np.arctan(np.abs(d_max) / (r_src + r_det))
angle_bot = np.arctan(np.abs(d_min) / (r_src + r_det))
xmin_top = np.abs(coarse_discr.max_pt[1])
xmin_bot = np.abs(coarse_discr.min_pt[1])
print((r_src - xmin_top) * np.sin(angle_top))
assert (r_src - xmin_top) * np.sin(angle_top) >= xmin_top * np.sqrt(2)
print((r_src - xmin_bot) * np.sin(angle_top))
assert (r_src - xmin_bot) * np.sin(angle_bot) >= xmin_bot * np.sqrt(2)
geometry = odl.tomo.FanFlatGeometry(angle_partition, det_partition,
                                    src_radius=r_src, det_radius=r_det)

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

background = coarse_discr.one()
masked_bg = coarse_mask(background)
masked_bg.show('masked background')
test_cube = fine_discr.one()

data = pspace_ray_trafo([background, test_cube])
data.show('data')

# %% Numerical detector ranges

# Adjust to reduce number of regions - tradeoff speed and data size
# num_div = angle_partition.shape[0]
# num_div = int(angle_partition.shape[0] / 10)

# TODO: make this a utility function
num_div = 2


def relevant_geometry_parts(angle_part, num_div, data):
    angle_div_step = int(np.ceil(angle_part.shape[0] / num_div))
    angle_parts = [angle_part[i * angle_div_step:(i + 1) * angle_div_step]
                   for i in range(num_div - 1)]
    if num_div >= 2:
        angle_parts.append(angle_partition[(num_div - 2) * angle_div_step:])

    data_arr = data.asarray()
    data_parts = [data_arr[i * angle_div_step:(i + 1) * angle_div_step]
                  for i in range(num_div - 1)]
    if num_div >= 2:
        data_parts.append(data_arr[(num_div - 2) * angle_div_step:])

    # Compute the minimum and maximum points of the nonzero regions
    nonzero_indcs = [np.nonzero(d) for d in data_parts]
    min_list, max_list, shape_list = [], [], []
    bdry_vecs = data.space.partition.cell_boundary_vecs[1:]
    for nz_idx in nonzero_indcs:
        xmin_j, xmax_j, shape_j = [], [], []
        for nz_idx_i, bvec_i in zip(nz_idx[1:], bdry_vecs):
            # Use the min/max along the angle coordinate to safely capture
            # all nonzero values
            imin, imax = np.min(nz_idx_i), np.max(nz_idx_i)
            xmin_j.append(bvec_i[imin])
            xmax_j.append(bvec_i[imax])
            shape_j.append(imax - imin)
        min_list.append(np.array(xmin_j))
        max_list.append(np.array(xmax_j))
        shape_list.append(np.array(shape_j, dtype=int))

    # TODO: use currently non-existing uniform_partition_frompartition
    det_parts = [odl.uniform_partition(xmin_j, xmax_j, shape_j)
                 for xmin_j, xmax_j, shape_j in zip(min_list, max_list,
                                                    shape_list)]

    return angle_parts, det_parts


angle_parts, det_parts = relevant_geometry_parts(angle_partition, num_div,
                                                 data)
geometries = [odl.tomo.Parallel2dGeometry(apart, dpart)
              for apart, dpart in zip(angle_parts, det_parts)]
ray_trafos = [odl.tomo.RayTransform(fine_discr, geom, impl='astra_cpu',
                                    det_pos_init=[20, 0])
              for geom in geometries]

# %%
