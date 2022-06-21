################################################################################
## I AM NOT THE AUTHOR OF THIS CODE, AUTHOR : https://github.com/silasbrack
## https://github.com/silasbrack/approximate-inference-for-bayesian-neural- 
## networks/blob/main/src/guides/radial.py 
## Using it for experimentation purposes
################################################################################

import numbers
from contextlib import ExitStack

import pyro
import pyro.distributions as dist
import pyro.infer.autoguide.initialization as ag_init
import torch
import tyxe
from pyro.distributions import constraints
from pyro.infer import autoguide
from torch.distributions import biject_to
from torch.distributions.utils import _standard_normal
from tyxe.guides import _get_base_dist


class Radial:
    def __init__(self):
        self.name = "Radial"

    def guide(self):
        return AutoRadial


class RadialNormal(dist.Normal):
    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(
            shape, dtype=self.loc.dtype, device=self.loc.device
        )
        distance = torch.randn(1, device=self.loc.device)
        normalizing_factor = torch.norm(eps, p=2)
        direction = eps / normalizing_factor
        eps_radial = direction * distance
        return self.loc + eps_radial * self.scale


class AutoRadial(autoguide.AutoGuide):
    def __init__(
        self,
        module,
        init_loc_fn=ag_init.init_to_median,
        init_scale=1e-1,
        train_loc=True,
        train_scale=True,
        max_guide_scale=None,
    ):
        module = ag_init.InitMessenger(init_loc_fn)(module)
        self.init_scale = init_scale
        self.train_loc = train_loc
        self.train_scale = train_scale
        self.max_guide_scale = max_guide_scale
        super().__init__(module)

    def _setup_prototype(self, *args, **kwargs):
        super()._setup_prototype(*args, **kwargs)

        for name, site in self.prototype_trace.iter_stochastic_nodes():
            constrained_value = site["value"]
            unconstrained_value = biject_to(site["fn"].support).inv(
                constrained_value
            )
            if self.train_loc:
                unconstrained_value = pyro.nn.PyroParam(unconstrained_value)
            autoguide.guides._deep_setattr(
                self, name + ".loc", unconstrained_value
            )
            if isinstance(self.init_scale, numbers.Real):
                scale_value = torch.full_like(site["value"], self.init_scale)
            elif isinstance(self.init_scale, str):
                scale_value = torch.full_like(
                    site["value"],
                    tyxe.util.calculate_prior_std(
                        self.init_scale, site["value"]
                    ),
                )
            else:
                scale_value = self.init_scale[site["name"]]
            scale_constraint = (
                constraints.positive
                if self.max_guide_scale is None
                else constraints.interval(0.0, self.max_guide_scale)
            )
            scale = (
                pyro.nn.PyroParam(scale_value, constraint=scale_constraint)
                if self.train_scale
                else scale_value
            )
            autoguide.guides._deep_setattr(self, name + ".scale", scale)

    def get_loc(self, site_name):
        return pyro.util.deep_getattr(self, site_name + ".loc")

    def get_scale(self, site_name):
        return pyro.util.deep_getattr(self, site_name + ".scale")

    def get_detached_distributions(self, site_names=None):
        if site_names is None:
            site_names = list(
                name
                for name, _ in self.prototype_trace.iter_stochastic_nodes()
            )

        result = dict()
        for name, site in self.prototype_trace.iter_stochastic_nodes():
            if name not in site_names:
                continue
            loc = self.get_loc(name).detach().clone()
            scale = self.get_scale(name).detach().clone()
            fn = RadialNormal(loc, scale).to_event(max(loc.dim(), scale.dim()))
            base_fn = _get_base_dist(site["fn"])
            if base_fn.support is not dist.constraints.real:
                fn = dist.TransformedDistribution(
                    fn, biject_to(base_fn.support)
                )
            result[name] = fn
        return result

    def forward(self, *args, **kwargs):
        if self.prototype_trace is None:
            self._setup_prototype(*args, **kwargs)

        plates = self._create_plates()
        result = {}
        for name, site in self.prototype_trace.iter_stochastic_nodes():
            with ExitStack() as stack:
                for frame in site["cond_indep_stack"]:
                    if frame.vectorized:
                        stack.enter_context(plates[frame.name])
                loc = self.get_loc(name)
                scale = self.get_scale(name)
                fn = RadialNormal(loc, scale).to_event(site["fn"].event_dim)
                base_fn = _get_base_dist(site["fn"])
                if base_fn.support is not dist.constraints.real:
                    fn = dist.TransformedDistribution(
                        fn, biject_to(base_fn.support)
                    )
                result[name] = pyro.sample(name, fn)
        return result