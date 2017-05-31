"""Multigrid volume alignment."""

from itertools import zip_longest
import numpy as np
import odl


# TODO: clean up and extend docs


class TranslationOperatorFixedTempl(odl.Operator):

    """Translation operator with translation vector as variable."""

    def __init__(self, f):
        """ Initialize a new instance.

        Parameters
        ----------
        f : DiscreteLpElement
            Fixed template in the translation.
        """
        assert isinstance(f, odl.DiscreteLpElement)
        self.f = f
        super().__init__(domain=odl.space.rn(f.space.ndim), range=f.space,
                         linear=False)

    def _call(self, t):
        """Translate ``self.f`` by ``t``.

        This produces the function :math:`x \\mapsto f(x - t)`.
        """
        mesh = self.f.space.meshgrid
        mesh_shifted = tuple(mi - ti for mi, ti in zip(mesh, t))
        return self.f.interpolation(mesh_shifted, bounds_check=False)


class TranslationCostFixedTempl(odl.solvers.Functional):

    """Functional to minimize in terms of the translation vector."""

    def __init__(self, cost, f, op=None, g=None):
        """Initialize a new instance.

        This functional evaluates the following expression, using the
        names of the variables to ``__init__``::

            J(t) = cost(op(f_t) - g)

        Here, ``f_t`` is f translated by ``t``.

        Parameters
        ----------
        cost : Functional
            Cost functional from ``op.range`` or ``f.space`` to the real
            numbers.
        f : DiscreteLpElement
            Fixed template in the translation.
        op : Operator, optional
            Operator with domain ``f.space``. ``None`` means identity
            operator on ``f.space``.
        g : ``op.range`` element-like, optional
            Shift of ``cost``. ``None`` means no shift.
        """
        assert isinstance(cost, odl.solvers.Functional)
        self.cost = cost
        assert isinstance(f, odl.DiscreteLpElement)
        self.f = f
        self.trans_op = TranslationOperatorFixedTempl(f)
        if op is None:
            self.op = odl.IdentityOperator(self.f.space)
        else:
            assert isinstance(op, odl.Operator)
            assert f in op.domain
            self.op = op
        if g is not None:
            assert g in self.op.range
        self.g = g

        super().__init__(space=self.trans_op.domain, linear=False)

    def _call(self, t):
        """Evaluate the cost at ``t``."""
        cost_arg = self.op(self.trans_op(t))
        if self.g is not None:
            cost_arg -= self.g

        return self.cost(cost_arg)

    @property
    def gradient(self):
        """Gradient operator of this functional."""

        func = self
        spatial_grad = odl.Gradient(func.f.space, pad_mode='order1')

        class TranslationCostFixedTemplGrad(odl.Operator):

            """Gradient operator of `TranslationCostFixedTempl`."""

            def __init__(self):
                """Initialize a new instance."""
                super().__init__(domain=func.domain, range=func.domain,
                                 linear=False)

            def _call(self, t):
                """Evaluate the gradient in ``t``."""
                # Translated f
                f_transl = func.trans_op(t)

                # Compute the cost gradient
                cost_arg = func.op(f_transl)
                if func.g is not None:
                    cost_arg -= func.g
                grad_cost = func.cost.gradient(cost_arg)

                # Apply derivative adjoint of `op` at the translated f
                # to the cost gradient. This is the left factor in the
                # inner product.
                factor_l = func.op.derivative(f_transl).adjoint(grad_cost)

                # Compute the right factors, consisting in grad(f_t)
                factors_r = spatial_grad(f_transl)

                # Take the inner products in f.space of factor_l and
                # the components of factors_r. The negative of this vector
                # is the desired result.
                return [-factor_l.inner(fac_r) for fac_r in factors_r]

        return TranslationCostFixedTemplGrad()


class TranslationCostFixedTemplNum(TranslationCostFixedTempl):

    @property
    def gradient(self):
        step = 2 * max(self.f.space.cell_sides)
        return odl.solvers.NumericalGradient(self, step=step)


# %% Testing

space = odl.uniform_discr([-1, -1], [1, 1], (150, 150), interp='linear')
templ = odl.phantom.shepp_logan(space, modified=True)
# Make a bit bigger to avoid hitting the boundary
resize = odl.ResizingOperator(space, ran_shp=(256, 256))
templ = resize(templ)
true_t = (0.5, -0.5)
templ_shifted = TranslationOperatorFixedTempl(templ)(true_t)
templ.show()
templ_shifted.show()

# Define spaces of different resolutions
space_hires = resize.range
space_midres = odl.uniform_discr(space_hires.min_pt, space_hires.max_pt,
                                 shape=(128, 128))
space_lowres = odl.uniform_discr(space_hires.min_pt, space_hires.max_pt,
                                 shape=(32, 32))

spaces = [space_lowres, space_midres, space_hires]

# Define geometry for ray transform
angle_partition = odl.uniform_partition(0, 2 * np.pi, 60)

# Make detector large enough to cover the object
src_radius = 10
det_radius = 10
opening_angle = np.arctan(
    max(np.max(np.abs(space_hires.min_pt)),
        np.max(np.abs(space_hires.max_pt))) *
    np.sqrt(3) / src_radius)
det_size = (np.floor(2 * (src_radius + det_radius) * np.sin(opening_angle)) *
            np.ones(1))
det_shape = (det_size / np.min(space_hires.cell_sides)).astype(int)
det_shape = 2 ** ((np.ceil(np.log2(det_shape))).astype(int))
det_max_pt = det_size / 2
det_min_pt = -det_max_pt
det_partition = odl.uniform_partition(det_min_pt, det_max_pt, det_shape)
geometry = odl.tomo.FanFlatGeometry(angle_partition, det_partition,
                                    src_radius, det_radius)

# Generate data
ray_trafo = odl.tomo.RayTransform(space_hires, geometry, impl='astra_cuda')
data = ray_trafo(templ_shifted)
noisy_data = (
    data +
    0.01 * np.max(data) * odl.phantom.white_noise(ray_trafo.range))

data_non_shifted = ray_trafo(templ)
data_non_shifted.show()


# %%
resampl = odl.Resampling(space_hires, space_lowres)
cur_templ = resampl(templ)
cur_templ_shifted = resampl(templ_shifted)

# Perform multi-resolution loop to optimize for the shift
bfgs_max_iter = 10
t = odl.rn(space_hires.ndim).zero()

with odl.util.Timer('Multigrid-BFGS:'):
    for cur_space, next_space in zip_longest(spaces, spaces[1:]):
        # Resample data by creating a ray transform with a coarser detector
        factor = space_hires.shape[0] // cur_space.shape[0]
        cur_det_shape = det_shape // factor
        cur_det_partition = odl.uniform_partition(det_min_pt, det_max_pt,
                                                  cur_det_shape)
        cur_geometry = odl.tomo.FanFlatGeometry(
            angle_partition, cur_det_partition, src_radius, det_radius)
        cur_ray_trafo = odl.tomo.RayTransform(cur_space, cur_geometry)

        resampl_data = odl.Resampling(data.space, cur_ray_trafo.range)
        cur_data = resampl_data(noisy_data)

        # Set up the translation cost
        transl_cost_num = TranslationCostFixedTemplNum(
            cost=odl.solvers.L2NormSquared(cur_ray_trafo.range),
            f=cur_templ,
            op=cur_ray_trafo,
            g=cur_data)

        # Define callback that shows the inversely shifted template with the
        # current translation guess
        fig = None

        def show_templ(t):
            global it
            global fig
            templ_transl = TranslationOperatorFixedTempl(cur_templ_shifted)(-t)
            fig = templ_transl.show(fig=fig)

        callback = (odl.solvers.CallbackPrintIteration() &
                    odl.solvers.CallbackPrint(fmt='t = {}') &
                    odl.solvers.CallbackApply(show_templ, step=1))
        callback = (odl.solvers.CallbackPrintIteration() &
                    odl.solvers.CallbackPrint(fmt='t = {}'))
        # Solve for the true translation
        try:
            line_search = odl.solvers.BacktrackingLineSearch(
                transl_cost_num, max_num_iter=8, discount=0.05)
            odl.solvers.bfgs_method(transl_cost_num, t,
                                    line_search=line_search,
                                    maxiter=bfgs_max_iter,
                                    callback=callback)
        except (ValueError, AssertionError):
            print('No more good BFGS updates')

        print('value after BFGS:', t)
        print('switching to gradient-free optimization')
        with odl.util.Timer('Coord descent:'):
            # Do coordinate descent, once per axis for a quick improvement
            num_eval = 10
            for axis in range(space.ndim):
                tvals = np.linspace(
                        t[axis] - 2 * cur_space.cell_sides[axis],
                        t[axis] + 2 * cur_space.cell_sides[axis],
                        num_eval)
                test_t = t.copy()
                func_vals = []
                for tval in tvals:
                    test_t[axis] = tval
                    func_vals.append(transl_cost_num(test_t))

                imin = int(np.argmin(func_vals))
                t[axis] = tvals[imin]

        print('value after coord descent:', t)

        # Resample template(s) to the space of the next iteration
        if next_space is not None:
            print('switching resolution')
            resampling = odl.Resampling(space_hires, next_space)
            cur_templ = resampling(templ)
            cur_templ_shifted = resampling(templ_shifted)
