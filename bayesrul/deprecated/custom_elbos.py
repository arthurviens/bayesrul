# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

from ast import Index
from pyexpat import model
import warnings
import weakref

import torch
from torch.distributions import kl_divergence
from torch.nn import GaussianNLLLoss

import pyro.ops.jit
from pyro.distributions.util import scale_and_mask
from pyro.infer.trace_elbo import Trace_ELBO
from pyro.infer.util import is_validation_enabled, check_fully_reparametrized
from pyro.infer.util import MultiFrameTensor, get_plate_stacks, torch_item
from pyro.infer.elbo import ELBO
from pyro.infer.enum import get_importance_trace
from pyro.util import check_if_enumerated, warn_if_nan


def _compute_log_r(model_trace, guide_trace):
    log_r = MultiFrameTensor()
    stacks = get_plate_stacks(model_trace)
    for name, model_site in model_trace.nodes.items():
        if model_site["type"] == "sample":
            log_r_term = model_site["log_prob"]
            if not model_site["is_observed"]:
                log_r_term = log_r_term - guide_trace.nodes[name]["log_prob"]
            log_r.add((stacks[name], log_r_term.detach()))
    return log_r

class CustomTrace_ELBO(Trace_ELBO):
    """
    A trace implementation of ELBO-based SVI. The estimator is constructed
    along the lines of references [1] and [2]. There are no restrictions on the
    dependency structure of the model or the guide. The gradient estimator includes
    partial Rao-Blackwellization for reducing the variance of the estimator when
    non-reparameterizable random variables are present. The Rao-Blackwellization is
    partial in that it only uses conditional independence information that is marked
    by :class:`~pyro.plate` contexts. For more fine-grained Rao-Blackwellization,
    see :class:`~pyro.infer.tracegraph_elbo.TraceGraph_ELBO`.

    References

    [1] Automated Variational Inference in Probabilistic Programming,
        David Wingate, Theo Weber

    [2] Black Box Variational Inference,
        Rajesh Ranganath, Sean Gerrish, David M. Blei
    """

    def _get_trace(self, model, guide, args, kwargs):
        #print("(trace elbo get trace) ^^' ")
        model_trace, guide_trace = super()._get_trace(
            model, guide, args, kwargs)
        return model_trace, guide_trace

    def loss(self, model, guide, *args, **kwargs):
        """
        :returns: returns an estimate of the ELBO
        :rtype: float

        Evaluates the ELBO with an estimator that uses num_particles many samples/particles.
        """
        loss = 0.0
        
        guide_trace = pyro.poutine.trace(guide).get_trace(*args, **kwargs)
        model_trace = pyro.poutine.trace(pyro.poutine.replay(model, guide_trace)).get_trace(
            *args, **kwargs
        )

        loss_particle, _ = self._differentiable_loss_particle(model_trace, guide_trace, args)
        loss = loss + loss_particle / self.num_particles

        warn_if_nan(loss, "loss")
        return loss

    def loss_and_grads(self, model, guide, *args, **kwargs):
        """
        :returns: returns an estimate of the ELBO
        :rtype: float

        Computes the ELBO as well as the surrogate ELBO that is used to form the gradient estimator.
        Performs backward on the latter. Num_particle many samples are used to form the estimators.
        """
        loss = 0.0
        
        # grab a trace from the generator
        guide_trace = pyro.poutine.trace(guide).get_trace(*args, **kwargs)
        model_trace = pyro.poutine.trace(pyro.poutine.replay(model, guide_trace)).get_trace(
            *args, **kwargs
        )

        loss_particle, surrogate_loss_particle = self._differentiable_loss_particle(model_trace, guide_trace, args)
        loss += loss_particle / self.num_particles

        # collect parameters to train from model and guide
        trainable_params = any(site["type"] == "param"
                                for trace in (model_trace, guide_trace)
                                for site in trace.nodes.values())

        if trainable_params and getattr(surrogate_loss_particle, 'requires_grad', False):
            surrogate_loss_particle = surrogate_loss_particle / self.num_particles
            surrogate_loss_particle.backward(retain_graph=self.retain_graph)
        warn_if_nan(loss, "loss")
        return loss

    def _differentiable_loss_particle(self, model_trace, guide_trace, args):
        elbo_particle = 0
        
        for name, model_site in model_trace.nodes.items():
            if model_site["type"] == "sample":
                if model_site["is_observed"]:
                    pass 
                    # Log Probs
                else:
                    if model_site["name"] == "likelihood.data_plate":
                        obs = args[1].view(-1, 1)
                        pred, var = model_site["value"].chunk(2, dim=-1)
                        assert pred.shape[0] == obs.shape[0], f"Output shape {pred.shape} should match {obs.shape}"
                        nll_loss = GaussianNLLLoss(reduction="sum")
                        elbo_particle = elbo_particle - nll_loss(pred, obs, var)

                    try:
                        guide_site = guide_trace.nodes[name]
                        if is_validation_enabled():
                            check_fully_reparametrized(guide_site)

                        # use kl divergence if available, else fall back on sampling
                        try:
                            kl_qp = kl_divergence(guide_site["fn"], model_site["fn"])
                            kl_qp = scale_and_mask(kl_qp, scale=guide_site["scale"], mask=guide_site["mask"])
                            if torch.is_tensor(kl_qp):
                                assert kl_qp.shape == guide_site["fn"].batch_shape
                                kl_qp_sum = kl_qp.sum()
                            else:
                                kl_qp_sum = kl_qp * torch.Size(guide_site["fn"].batch_shape).numel()
                            elbo_particle = elbo_particle - kl_qp_sum
                        except NotImplementedError: # Doesn't go dere
                            entropy_term = guide_site["score_parts"].entropy_term
                            elbo_particle = elbo_particle + model_site["log_prob_sum"] - entropy_term.sum()
                    except KeyError:
                        pass

        # handle auxiliary sites in the guide
        for name, guide_site in guide_trace.nodes.items():
            if guide_site["type"] == "sample" and name not in model_trace.nodes:
                assert guide_site["infer"].get("is_auxiliary")
                if is_validation_enabled():
                    check_fully_reparametrized(guide_site)
                entropy_term = guide_site["score_parts"].entropy_term
                elbo_particle = elbo_particle - entropy_term.sum()


        loss = -(elbo_particle.detach() if torch._C._get_tracing_state() else torch_item(elbo_particle))
        surrogate_loss = -elbo_particle
        return loss, surrogate_loss
