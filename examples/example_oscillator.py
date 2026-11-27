"""A toy back-end for DBE-P and a consumer.

Dynamics: forced isotropic oscillator  r'' = -k r + g cos(w t) e_x,  with an impulse dv at the start of
the step. Natively the state is measured from a centre "ORIGIN" that moves on a circle relative to "SSB".

The consumer states in the model specification what it wants z to be (a StateRequest: centre and axes).
The back-end converts internally when its native centre differs, and its Jacobians are those of the
requested state, including the terms that come from the centre's motion. A request it cannot meet is refused
when the model is built.

Derivatives come from forward-mode autodiff through a fixed-step RK4 map, standing in for a back-end
wrapped by an adapter. Parameter k is differentiated, g is held fixed: J["p"] has only "k".
"""
import jax
import jax.numpy as jnp
import numpy as np

import dbep
from dbep import (Capabilities, Control, DBEPUnsupported, DenseResult, Descriptor, Param, Requires, Slice,
                  SliceRequest, StateRequest, StepResult, TimeConvention)

jax.config.update("jax_enable_x64", True)

W, AMP, N_RK = 0.3, 2.0, 1000
EPHEMERIS_ID = "toy-circle"
NATIVE_CENTRE = "ORIGIN"
SUPPORTED_CENTRES = ("ORIGIN", "SSB")


def _origin_state(t):
    """State of ORIGIN relative to SSB: position and velocity (6,)."""
    c = AMP * jnp.array([jnp.cos(W * t), jnp.sin(W * t), 0.0])
    cv = AMP * W * jnp.array([-jnp.sin(W * t), jnp.cos(W * t), 0.0])
    return jnp.concatenate([c, cv])


def _offset(t, centre):
    """Native (ORIGIN-centred) -> requested centre: add this vector."""
    return jnp.zeros(6) if centre == "ORIGIN" else _origin_state(t)


def _flow(inp):
    z, u, t, dt, p = inp["z"], inp["u"], inp["t"], inp["dt"], inp["p"]
    x = z.at[3:].add(u["dv"])                       # impulse at the start of the step
    h = dt / N_RK
    ex = jnp.array([1.0, 0.0, 0.0])

    def f(tt, s):
        return jnp.concatenate([s[3:], -p["k"] * s[:3] + p["g"] * jnp.cos(W * tt) * ex])

    def body(i, s):
        tt = t + i * h
        k1 = f(tt, s)
        k2 = f(tt + h / 2, s + h / 2 * k1)
        k3 = f(tt + h / 2, s + h / 2 * k2)
        k4 = f(tt + h, s + h * k3)
        return s + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return jax.lax.fori_loop(0, N_RK, body, x)


def _flow_ext(inp, centre):
    """The map in the requested coordinates: convert in, integrate natively, convert out."""
    z_native = inp["z"] - _offset(inp["t"], centre)
    z1 = _flow({**inp, "z": z_native})
    return z1 + _offset(inp["t"] + inp["dt"], centre)


class OscillatorBackend:
    DEFAULTS = {"k": 1.0, "g": 0.0}

    def __init__(self, differentiate=("k",), state: StateRequest = None):
        centre = NATIVE_CENTRE
        for name in ("r", "v"):
            sr = state.for_slice(name) if state else None
            if sr is None:
                continue
            if sr.axes not in ("", "ICRF"):
                raise DBEPUnsupported(f"axes {sr.axes!r} not supported (only ICRF)")
            if sr.unit not in ("", "km" if name == "r" else "km/s"):
                raise DBEPUnsupported(f"unit {sr.unit!r} not supported for {name!r}")
            if sr.centre:
                if sr.centre not in SUPPORTED_CENTRES:
                    raise DBEPUnsupported(f"centre {sr.centre!r} not supported (have {SUPPORTED_CENTRES})")
                if centre != NATIVE_CENTRE and sr.centre != centre:
                    raise DBEPUnsupported("r and v must share one centre")
                centre = sr.centre
        self._centre = centre
        self._diff = tuple(differentiate)
        self._d = Descriptor(
            profile="orbital/1",                    # a toy; declared so the consumer's check passes
            time=TimeConvention("TDB", "J2000", "s"),
            state=(Slice("r", 3, "position", "km", "ICRF", centre),
                   Slice("v", 3, "velocity", "km/s", "ICRF", centre)),
            controls=(Control("dv", (3,), "delta_v", "km/s"),),
            params=(Param("k", (), 1.0, "1/s^2", "k" in self._diff),
                    Param("g", (), 0.0, "km/s^2", "g" in self._diff)),
            capabilities=Capabilities(level=2, dense=True, batch=False, jit=True, reverse=True),
            ephemeris=EPHEMERIS_ID,
        )
        self._meta = self._d.state_meta()

    def describe(self) -> Descriptor:
        return self._d

    # -- helpers
    @staticmethod
    def _f64(tree):
        return jax.tree_util.tree_map(lambda a: jnp.asarray(a, dtype=jnp.float64), tree)

    def _split(self, p):
        pp = self._f64({**self.DEFAULTS, **p})
        return ({k: v for k, v in pp.items() if k in self._diff},
                {k: v for k, v in pp.items() if k not in self._diff})

    def _flow_and_jac(self, z, u, t, dt, p_diff, p_fixed):
        def g(d):
            return _flow_ext({**d, "p": {**p_fixed, **d["p"]}}, self._centre)

        d = self._f64({"z": z, "u": u, "t": t, "dt": dt, "p": p_diff})
        return g(d), jax.jacfwd(g)(d)

    # -- protocol
    def step(self, z, u, t, dt, p):
        u = {**{"dv": jnp.zeros(3)}, **u}
        p_diff, p_fixed = self._split(p)
        z1, J = self._flow_and_jac(z, u, t, dt, p_diff, p_fixed)

        def dense(t_vec, partials=False):
            tv = jnp.asarray(t_vec, dtype=jnp.float64)
            t0 = jnp.asarray(t, dtype=jnp.float64)
            if not partials:                                # cheap path: states only, no variational work
                zs = jax.vmap(lambda s: _flow_ext(self._f64({"z": z, "u": u, "t": t, "dt": s - t0,
                                                             "p": {**p_diff, **p_fixed}}), self._centre))(tv)
                return DenseResult(z=zs, meta=self._meta, J=None)
            zs, Js = jax.vmap(lambda s: self._flow_and_jac(z, u, t, s - t0, p_diff, p_fixed))(tv)
            return DenseResult(z=zs, meta=self._meta, J=Js)

        return StepResult(z=z1, J=J, status=jnp.int32(0), meta=self._meta, _dense=dense)


# ----------------------------------------------------------------------------- a consumer
def demo():
    # The consumer's model specification: position and velocity wanted relative to SSB, in ICRF axes.
    want = StateRequest(slices=(SliceRequest("r", centre="SSB", axes="ICRF", unit="km"),
                                SliceRequest("v", centre="SSB", axes="ICRF", unit="km/s")))
    b = OscillatorBackend(differentiate=("k",), state=want)
    d = b.describe()

    # 1. requirement check, including the coordinates (what a consumer does when a model is built)
    dbep.check(Requires(profile="orbital/1", level=2, roles=("position", "velocity"), state=want, dense=True), d)

    # 2. conformance suite, on the converted model
    z0 = np.array([1.0, 0.0, 0.2, 0.0, 0.5, 0.0])
    u0 = {"dv": np.array([0.0, 0.1, 0.0])}
    p = {"k": 0.8, "g": 0.05}
    dbep.check_meta(b, z0, u0, 0.0, 10.0, p)
    errs = dbep.check_step_jacobians(b, z0, u0, 0.0, 10.0, p)
    print("meta consistent; J tree leaves:", sorted(errs))
    print("step Jacobian FD errors:", {k: f"{v:.1e}" for k, v in errs.items()})
    print("dense consistency:", {k: f"{v:.1e}" for k, v in dbep.check_dense(b, z0, u0, 0.0, 10.0, p).items()})
    print("chaining:", {k: f"{v:.1e}" for k, v in dbep.check_chaining(b, z0, u0, 0.0, 10.0, 20.0, p).items()})

    # 3. the same physics requested natively differs by exactly the centre's state, and the converted
    #    Jacobians differ in the time blocks (the centre moves) but not in the state block
    native = OscillatorBackend(differentiate=("k",))
    t0, dt = 0.0, 10.0
    zn0 = np.asarray(z0 - _origin_state(t0))
    ra = native.step(zn0, u0, t0, dt, p)
    rb = b.step(z0, u0, t0, dt, p)
    print("converted - native state, minus centre state:",
          f"{np.abs(np.asarray(rb.z) - np.asarray(ra.z) - np.asarray(_origin_state(t0 + dt))).max():.1e}")
    print("J['z'] converted vs native:", f"{np.abs(np.asarray(rb.J['z']) - np.asarray(ra.J['z'])).max():.1e}")
    print("J['dt'] differs by centre velocity/acceleration term:",
          f"{np.abs(np.asarray(rb.J['dt']) - np.asarray(ra.J['dt'])).max():.2f}")

    # 4. consumer: two segments, observations inside segment 2. No ephemeris is needed here: z is already
    #    what the consumer asked for, so the geometry is a selection of z.
    u1, u2 = u0, dbep.zero_controls(d)
    dt1, dt2 = 10.0, 20.0
    r1 = b.step(z0, u1, t0, dt1, p)
    r2 = b.step(r1.z, u2, t0 + dt1, dt2, p)
    t_obs = t0 + dt1 + np.array([5.0, 10.0, 15.0, 20.0])
    dn = r2.dense(t_obs, partials=True)                     # relative to the start of segment 2
    sl = dn.meta.index("r")
    sel = np.zeros((3, dn.meta.n_z))
    sel[:, sl] = np.eye(3)                                  # dy/dz, y = position relative to SSB

    Phi1 = np.asarray(r1.J["z"])
    S1 = np.asarray(r1.J["p"]["k"])
    H_z0 = np.einsum("ij,tjk->tik", sel, np.asarray(dn.J["z"]) @ Phi1)
    H_k = np.einsum("ij,tj->ti", sel, np.einsum("tij,j->ti", np.asarray(dn.J["z"]), S1) + np.asarray(dn.J["p"]["k"]))

    def y_full(z, pp):                                      # reference: the whole chain, plain evaluation
        a = b.step(z, u1, t0, dt1, pp)
        c = b.step(a.z, u2, t0 + dt1, dt2, pp)
        return np.asarray(c.dense(t_obs).z)[:, sl]          # states only: the cheap path

    h = 1e-6
    fd_z0 = np.stack([(y_full(z0 + h * e, p) - y_full(z0 - h * e, p)) / (2 * h) for e in np.eye(6)], axis=-1)
    fd_k = (y_full(z0, {**p, "k": p["k"] + h}) - y_full(z0, {**p, "k": p["k"] - h})) / (2 * h)
    print(f"H wrt z0 : max |chain - FD| = {np.abs(H_z0 - fd_z0).max():.1e}")
    print(f"H wrt k  : max |chain - FD| = {np.abs(H_k - fd_k).max():.1e}")

    # 4b. derivative with respect to a boundary epoch T1 (end of segment 1 = start of segment 2, impulse at T1),
    #     observations at fixed epochs: DBE-P section 12, rule 7
    H_T1 = np.einsum("ij,tj->ti", sel, np.einsum("tij,j->ti", np.asarray(dn.J["z"]), np.asarray(r1.J["dt"]))
                     + np.asarray(dn.J["t"]) - np.asarray(dn.J["dt"]))

    def y_T1(T1):
        a = b.step(z0, u1, t0, T1 - t0, p)
        c = b.step(a.z, u2, T1, dt2, p)
        return np.asarray(c.dense(t_obs).z)[:, sl]

    fd_T1 = (y_T1(t0 + dt1 + h) - y_T1(t0 + dt1 - h)) / (2 * h)
    print(f"H wrt boundary epoch T1 : max |chain - FD| = {np.abs(H_T1 - fd_T1).max():.1e}")

    # 5. a request the back-end cannot meet is refused when the model is built
    try:
        OscillatorBackend(state=StateRequest(slices=(SliceRequest("r", centre="MARS"),)))
    except DBEPUnsupported as e:
        print("refused at construction:", e)

    # 6. and a model whose realised state does not match the consumer's request fails the check
    try:
        dbep.check(Requires(profile="orbital/1", state=want), native.describe())
    except dbep.DBEPMismatch as e:
        print("check on a native-centre model:", e)


if __name__ == "__main__":
    demo()
