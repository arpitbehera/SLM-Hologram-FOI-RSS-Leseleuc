"""CGH design: minimise the FOI (or RSS) cost over the nearfield phase.

Three backends:

* ``"scipy"`` (default, **faithful** to the paper): true nonlinear conjugate
  gradient via :func:`scipy.optimize.minimize(method="CG")` using the analytic
  gradient (Hamamatsu p5). Runs on CPU, deterministic per seed.
* ``"torch"``: autodiff gradient descent (Adam/LBFGS). Mirrors what slmsuite's
  experimental ``method="CG"`` does internally; useful for GPU runs.
* ``"slmsuite"``: drive ``slmsuite.holography.algorithms.Hologram.optimize(
  method="CG", loss=FOILoss())`` directly. Experimental; GPU-oriented.

Note: slmsuite's ``"CG"`` is gradient descent (Adam by default), not strict
conjugate gradient, so for an exact CGM reproduction prefer ``"scipy"``.
"""

from __future__ import annotations

import inspect

import numpy as np
from scipy.optimize import minimize

from .losses import COST_GRAD, rss_cost_grad, foi_cost_grad

# Scale-aware solver tolerances (see docs/INSIGHTS.md "Large-mask numerical
# stability"). The RSS gradient magnitude shrinks with the farfield grid size
# M = n*oversample (||grad||_2 ~ 1/n^2, max|grad| ~ 1/n^3) because the unit total
# intensity is spread over M^2 pixels. Anchoring tolerances to the *initial*
# gradient instead of an absolute constant keeps convergence size-independent.
CG_GTOL_REL = 1e-6        # CG stops once ||grad||_inf drops this far below its start
ADAM_EPS = 1e-16          # Adam denominator floor; << any large-mask gradient (was 1e-8)


def _relative_gtol(g0_inf, rel=CG_GTOL_REL, floor=1e-300):
    """Scale-aware conjugate-gradient stopping tolerance.

    scipy CG terminates when ``max|grad| < gtol``. A fixed absolute ``gtol`` (the
    old code used ``1e-9``) is blind to problem size: for an SLM mask of n >~ 300
    the RSS gradient *starts* below 1e-9, so CG reports convergence at iteration 0
    and returns the initial random phase (a blank hologram). Anchoring ``gtol`` to
    the initial gradient infinity-norm ``g0_inf`` makes the rule scale-free -- stop
    once the gradient has fallen ``rel`` (~6 orders) below where it began -- while
    ``floor`` keeps a vanishing initial gradient from yielding a zero tolerance.
    """
    return max(float(g0_inf), floor) * rel


def _normalize_target_natural(T_centered, method, xp=np):
    """Convert a centered target intensity to natural ordering with the right norm."""
    T_nat = xp.fft.ifftshift(T_centered)
    if method == "RSS":
        # RSS needs sum-normalised target (matches sum_i I_i == 1).
        s = xp.sum(T_nat)
        if s > 0:
            T_nat = T_nat / s
    # FOI uses its own L2 normalisation internally; leave magnitude as-is.
    return T_nat


def initial_phase(n, seed, xp=np):
    """Uniform random phase in [0, 2pi) (paper's initialisation)."""
    rng = np.random.default_rng(seed)
    phase = rng.uniform(0.0, 2 * np.pi, size=(n, n))
    return xp.asarray(phase)


def design_cgh(
    target_centered,
    amp,
    oversample,
    method="FOI",
    seed=0,
    phase0=None,
    iters=1000,
    backend="scipy",
    xp=np,
    return_info=False,
    **backend_kwargs,
):
    """Design a phase-only CGH for ``target_centered`` (centered intensity).

    Returns the optimized nearfield phase (n x n), wrapped to [0, 2pi).
    If ``return_info``, also returns a dict with cost history / solver result.
    """
    if method not in COST_GRAD:
        raise ValueError(f"unknown method {method!r}; use 'FOI' or 'RSS'.")
    n = amp.shape[0]
    T_nat = _normalize_target_natural(target_centered, method, xp=xp)
    if phase0 is None:
        phase0 = initial_phase(n, seed, xp=xp)
    else:
        phase0 = xp.asarray(phase0)

    if backend == "scipy":
        phase, info = _design_scipy(phase0, amp, T_nat, oversample, method, iters)
    elif backend == "torch":
        phase, info = _design_torch(phase0, amp, T_nat, oversample, method, iters, **backend_kwargs)
    elif backend == "slmsuite":
        phase, info = _design_slmsuite(target_centered, amp, oversample, method,
                                       seed, iters, phase0=phase0, **backend_kwargs)
    else:
        raise ValueError(f"unknown backend {backend!r}.")

    phase = np.mod(phase, 2 * np.pi)
    if return_info:
        return phase, info
    return phase


def _design_scipy(phase0, amp, T_nat, oversample, method, iters):
    n = amp.shape[0]
    cost_grad = COST_GRAD[method]
    history = []

    def obj(x):
        f, g = cost_grad(x.reshape(n, n), amp, T_nat, oversample, xp=np)
        return float(f), np.asarray(g, dtype=np.float64).ravel()

    def cb(xk):
        history.append(obj(xk)[0])

    x0 = np.asarray(phase0, dtype=np.float64).ravel()
    g0_inf = float(np.abs(obj(x0)[1]).max())
    gtol = _relative_gtol(g0_inf)
    res = minimize(
        obj,
        x0,
        jac=True,
        method="CG",
        callback=cb,
        options={"maxiter": int(iters), "gtol": gtol},
    )
    phase = res.x.reshape(n, n)
    info = {"final_cost": float(res.fun), "nit": int(res.nit), "history": history,
            "result": res, "gtol": gtol, "g0_inf": g0_inf}
    return phase, info


def design_cgh_dual(
    target_centered, amp, oversample, method="FOI", seed=0, iters=1000, phase0=None,
):
    """Scipy CG design of ``method`` that records BOTH RSS and FOI cost per iter.

    The optimizer minimizes ``method``'s objective (on that method's own target
    normalization), while the callback evaluates both cost functions every
    iteration as diagnostics. Returns ``(phase, info)`` with ``info`` keys:
    ``final_cost`` (active method), ``nit``, ``rss_history``, ``foi_history``
    (equal-length lists), ``method``.

    scipy-only: it relies on the CG callback for per-iteration history.
    """
    if method not in COST_GRAD:
        raise ValueError(f"unknown method {method!r}; use 'FOI' or 'RSS'.")
    n = amp.shape[0]
    T_rss = _normalize_target_natural(target_centered, "RSS")
    T_foi = _normalize_target_natural(target_centered, "FOI")
    T_active = T_rss if method == "RSS" else T_foi
    if phase0 is None:
        phase0 = initial_phase(n, seed)
    phase0 = np.asarray(phase0, dtype=np.float64)

    cost_grad = COST_GRAD[method]
    rss_history, foi_history = [], []

    def obj(x):
        f, g = cost_grad(x.reshape(n, n), amp, T_active, oversample, xp=np)
        return float(f), np.asarray(g, dtype=np.float64).ravel()

    def cb(xk):
        p = xk.reshape(n, n)
        rss_history.append(float(rss_cost_grad(p, amp, T_rss, oversample, xp=np)[0]))
        foi_history.append(float(foi_cost_grad(p, amp, T_foi, oversample, xp=np)[0]))

    x0 = phase0.ravel()
    gtol = _relative_gtol(np.abs(obj(x0)[1]).max())
    res = minimize(
        obj, x0, jac=True, method="CG", callback=cb,
        options={"maxiter": int(iters), "gtol": gtol},
    )
    phase = np.mod(res.x.reshape(n, n), 2 * np.pi)
    info = {
        "final_cost": float(res.fun), "nit": int(res.nit),
        "rss_history": rss_history, "foi_history": foi_history, "method": method,
    }
    return phase, info


def _design_torch(phase0, amp, T_nat, oversample, method, iters, lr=0.1, optimizer="Adam",
                  adam_eps=ADAM_EPS, **_):
    import torch

    from .forward import _embed, _crop  # noqa: F401  (kept for parity / debugging)

    n = amp.shape[0]
    m = n * oversample
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    amp_t = torch.as_tensor(np.asarray(amp), dtype=torch.float64, device=dev)
    T_t = torch.as_tensor(np.asarray(T_nat), dtype=torch.float64, device=dev)
    phase_t = torch.as_tensor(np.asarray(phase0), dtype=torch.float64, device=dev).clone().requires_grad_(True)

    off = (m - n) // 2

    def loss_fn():
        psi = amp_t * torch.exp(1j * phase_t)
        P = torch.zeros((m, m), dtype=torch.complex128, device=dev)
        P[off:off + n, off:off + n] = psi
        Psi = torch.fft.fft2(P, norm="ortho")
        I = torch.abs(Psi) ** 2
        if method == "FOI":
            Itil = I / torch.linalg.vector_norm(I)
            Ttil = T_t / torch.linalg.vector_norm(T_t)
            return -torch.sum(Itil * Ttil)
        else:  # RSS
            return torch.sum((I - T_t) ** 2)

    opt_class = getattr(torch.optim, optimizer)
    if optimizer == "LBFGS":
        opt = opt_class([phase_t], lr=lr, max_iter=int(iters))

        def closure():
            opt.zero_grad()
            l = loss_fn()
            l.backward()
            return l

        opt.step(closure)
        final = float(loss_fn().detach())
    else:
        # Adam-family optimizers gate the step by sqrt(v)+eps. The default
        # eps=1e-8 is tuned for O(1) ML losses; the large-mask RSS gradient sinks
        # to ~1e-11, where eps swamps sqrt(v) and Adam loses its scale-invariance.
        # Float64 lets us push eps to the noise floor and restore it.
        opt_kwargs = {"lr": lr}
        if "eps" in inspect.signature(opt_class).parameters:
            opt_kwargs["eps"] = adam_eps
        opt = opt_class([phase_t], **opt_kwargs)
        final = None
        for _i in range(int(iters)):
            opt.zero_grad()
            l = loss_fn()
            l.backward()
            opt.step()
            final = float(l.detach())

    phase = phase_t.detach().cpu().numpy()
    return phase, {"final_cost": final}


def _design_slmsuite(target_centered, amp, oversample, method, seed, iters,
                     phase0=None, optimizer="Adam", lr=0.1, adam_eps=ADAM_EPS, **_):
    """Best-effort adapter onto slmsuite's experimental CG path (GPU-oriented).

    slmsuite's ``optimize_cg`` calls ``optimizer.step()`` without a closure, so
    closure-only optimizers (LBFGS) are unsupported here; Adam is the default and
    matches slmsuite's own CG configuration.
    """
    from slmsuite.holography.algorithms import Hologram
    from .losses import TORCH_LOSSES

    n = amp.shape[0]
    m = n * oversample
    # slmsuite target is an amplitude on the (centered) farfield grid of shape m x m.
    target_amp = np.sqrt(np.asarray(target_centered, dtype=np.float64))
    target_amp = np.pad(  # ensure m x m
        target_amp,
        [(0, m - target_amp.shape[0]), (0, m - target_amp.shape[1])],
        mode="constant",
    ) if target_amp.shape[0] < m else target_amp

    if phase0 is None:
        phase0 = initial_phase(n, seed)
    phase0 = np.asarray(phase0, dtype=np.float64)
    holo = Hologram(target=target_amp, amp=np.asarray(amp), phase=phase0, slm_shape=(n, n), dtype=np.float64)
    # Scale-aware Adam eps (see _design_torch): keep the eps gate below the
    # large-mask RSS gradient so Adam stays scale-invariant.
    opt_kwargs = {"lr": lr}
    if optimizer.startswith("Adam") or optimizer in ("NAdam", "RAdam", "Adamax"):
        opt_kwargs["eps"] = adam_eps
    holo.optimize(
        method="CG",
        maxiter=int(iters),
        loss=TORCH_LOSSES[method](),
        optimizer=optimizer,
        optimizer_kwargs=opt_kwargs,
        verbose=False,
    )
    holo_phase = holo.phase
    if hasattr(holo_phase, "get"):  # cupy array on GPU backend
        holo_phase = holo_phase.get()
    phase = np.asarray(holo_phase)
    return phase, {"final_cost": holo.flags.get("loss_result")}
