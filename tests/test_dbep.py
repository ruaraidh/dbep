"""Tests for the DBE-P toolkit: the conversion wrapper, the orientation service and the extended requirement check."""
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import dbep
from dbep import (Capabilities, Control, Converted, DBEPMismatch, DBEPUnsupported, Descriptor, IAUOrientation, Param,
                  Requires, Slice, SliceRequest, StateRequest, StepResult, TimeConvention)
import example_oscillator as eo

jax.config.update("jax_enable_x64", True)
WANT_SSB = StateRequest(slices=(SliceRequest("r", centre="SSB", axes="ICRF", unit="km"),
                                SliceRequest("v", centre="SSB", axes="ICRF", unit="km/s")))


class CircleEphemeris:
    """SSB-relative states of ORIGIN (on a circle, as in the toy) and of a body PLANET on another circle."""
    id = "toy-circle"

    def state(self, body, t):
        if body == "ORIGIN":
            return eo._origin_state(t)
        if body == "PLANET":
            w, a = 0.11, 7.0
            return a * jnp.array([jnp.cos(w * t), jnp.sin(w * t), 0.2 * jnp.sin(w * t), -w * jnp.sin(w * t), w * jnp.cos(w * t), 0.2 * w * jnp.cos(w * t)])
        raise KeyError(body)


class TwoOsc:
    """Native state: spacecraft (r, v) and an asteroid (r:AST, v:AST), both relative to ORIGIN; independent springs."""
    N_RK = 400

    def __init__(self):
        sl = (Slice("r", 3, "position", "km", "ICRF", "ORIGIN"), Slice("v", 3, "velocity", "km/s", "ICRF", "ORIGIN"),
              Slice("r:AST", 3, "position", "km", "ICRF", "ORIGIN"), Slice("v:AST", 3, "velocity", "km/s", "ICRF", "ORIGIN"))
        self._d = Descriptor("orbital/1", TimeConvention(), sl, (Control("dv", (3,), "delta_v", "km/s"),),
                             (Param("k", (), 1.0, "1/s^2", True),), Capabilities(level=2, dense=True, jit=True), ephemeris="")
        self._meta = self._d.state_meta()

    def describe(self):
        return self._d

    def _flow(self, inp):
        z, u, t, dt, p = inp["z"], inp["u"], inp["t"], inp["dt"], inp["p"]
        x = z.at[3:6].add(u["dv"])
        h = dt / self.N_RK
        # the state is ordered [r, v, r:AST, v:AST]: derivatives r' = v, v' = ..., ra' = va, va' = ...
        def g(tt, s):
            return jnp.concatenate([s[3:6], -p["k"] * s[0:3] + 0.01 * jnp.cos(0.3 * tt) * jnp.array([1., 0, 0]), s[9:12], -0.4 * s[6:9]])

        def body(i, s):
            tt = t + i * h
            k1 = g(tt, s); k2 = g(tt + h / 2, s + h / 2 * k1); k3 = g(tt + h / 2, s + h / 2 * k2); k4 = g(tt + h, s + h * k3)
            return s + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return jax.lax.fori_loop(0, self.N_RK, body, x)

    def step(self, z, u, t, dt, p):
        u = {"dv": jnp.zeros(3), **u}
        pp = {"k": jnp.asarray(p.get("k", 1.0), dtype=jnp.float64)}
        d = {"z": jnp.asarray(z, dtype=jnp.float64), "u": {"dv": jnp.asarray(u["dv"], dtype=jnp.float64)},
             "t": jnp.asarray(t, dtype=jnp.float64), "dt": jnp.asarray(dt, dtype=jnp.float64), "p": pp}
        z1, J = self._flow(d), jax.jacfwd(self._flow)(d)

        def dense(t_vec, partials=False):
            tv = jnp.asarray(t_vec, dtype=jnp.float64)
            def one(s):
                dd = {**d, "dt": s - d["t"]}
                return self._flow(dd), jax.jacfwd(self._flow)(dd)
            zs, Js = jax.vmap(one)(tv)
            return dbep.DenseResult(z=zs, meta=self._meta, J=Js if partials else None)
        return StepResult(z=z1, J=J, status=jnp.int32(0), meta=self._meta, _dense=dense)


Z0 = np.array([1.0, 0.0, 0.2, 0.0, 0.5, 0.0])
U0 = {"dv": np.array([0.0, 0.1, 0.0])}
P0 = {"k": 0.8, "g": 0.05}


def test_wrapper_matches_builtin_conversion():
    """The wrapper around the native toy (centre ORIGIN) equals the toy that converts internally."""
    native = eo.OscillatorBackend(differentiate=("k",))
    wrapped = Converted(native, WANT_SSB, CircleEphemeris())
    builtin = eo.OscillatorBackend(differentiate=("k",), state=WANT_SSB)
    dbep.check(Requires(profile="orbital/1", level=2, roles=("position", "velocity"), state=WANT_SSB, dense=True), wrapped.describe())
    a, b = wrapped.step(Z0, U0, 0.0, 10.0, P0), builtin.step(Z0, U0, 0.0, 10.0, P0)
    assert np.max(np.abs(a.z - b.z)) < 1e-12
    for path, leaf in dbep.leaves(b.J):
        assert np.max(np.abs(np.asarray(leaf) - np.asarray(dbep.get(a.J, path)))) < 1e-9, path
    tv = np.array([2.0, 5.0, 10.0])
    da, db = a.dense(tv, partials=True), b.dense(tv, partials=True)
    assert np.max(np.abs(da.z - db.z)) < 1e-12
    for path, leaf in dbep.leaves(db.J):
        assert np.max(np.abs(np.asarray(leaf) - np.asarray(dbep.get(da.J, path)))) < 1e-9, path
    assert a.dense(tv).J is None and a.meta == wrapped.describe().state_meta()
    assert "toy-circle" in wrapped.describe().ephemeris


def test_wrapper_passes_conformance():
    wrapped = Converted(eo.OscillatorBackend(differentiate=("k",)), WANT_SSB, CircleEphemeris())
    dbep.check_meta(wrapped, Z0, U0, 0.0, 10.0, P0)
    dbep.check_step_jacobians(wrapped, Z0, U0, 0.0, 10.0, P0)
    dbep.check_dense(wrapped, Z0, U0, 0.0, 10.0, P0)
    dbep.check_chaining(wrapped, Z0, U0, 0.0, 10.0, 20.0, P0)


def test_entity_in_z_as_centre():
    """The centre is the integrated asteroid: T depends on z, so J['z'] changes. Checked by finite differences and by value."""
    want = StateRequest(slices=(SliceRequest("r", centre="AST", axes="ICRF", unit="km"), SliceRequest("v", centre="AST")))
    native = TwoOsc()
    w = Converted(native, want, CircleEphemeris())
    assert [s.centre for s in w.describe().state] == ["AST", "AST", "ORIGIN", "ORIGIN"]
    z0 = np.array([1.0, 0.0, 0.2, 0.0, 0.5, 0.0, 0.3, -0.2, 0.1, 0.05, 0.02, -0.01])
    u, p = {"dv": np.array([0.0, 0.1, 0.0])}, {"k": 0.8}
    res = w.step(z0, u, 1.0, 10.0, p)
    zn0 = z0.copy(); zn0[:6] = z0[:6] + np.concatenate([z0[6:9], z0[9:12]])       # requested (rel AST) -> native (rel ORIGIN)
    rn = native.step(zn0, u, 1.0, 10.0, p)
    expect = np.asarray(rn.z).copy(); expect[:6] -= np.asarray(rn.z)[6:12]
    assert np.max(np.abs(np.asarray(res.z) - expect)) < 1e-12
    assert np.max(np.abs(np.asarray(res.J["z"]) - np.asarray(rn.J["z"]))) > 1e-3       # the state Jacobian is not the native one
    dbep.check_meta(w, z0, u, 1.0, 10.0, p)
    dbep.check_step_jacobians(w, z0, u, 1.0, 10.0, p, tol=1e-6)
    dbep.check_dense(w, z0, u, 1.0, 10.0, p)
    dbep.check_chaining(w, z0, u, 1.0, 10.0, 20.0, p)


def test_planet_centre_is_exercised():
    """Request relative to a body that moves on its own circle; the time terms enter J['t'] and J['dt']."""
    want = StateRequest(slices=(SliceRequest("r", centre="PLANET"), SliceRequest("v", centre="PLANET")))
    w = Converted(eo.OscillatorBackend(differentiate=("k",)), want, CircleEphemeris())
    dbep.check_step_jacobians(w, Z0, U0, 3.0, 10.0, P0)
    dbep.check_dense(w, Z0, U0, 3.0, 10.0, P0)


def test_refusals():
    nat = eo.OscillatorBackend(differentiate=("k",))
    for slices in ((SliceRequest("r", centre="MARS"),), (SliceRequest("r", axes="EME2000"),), (SliceRequest("r", unit="m"),),
                   (SliceRequest("q"),), (SliceRequest("r", centre="SSB"), SliceRequest("v", centre="PLANET"))):
        with pytest.raises(DBEPUnsupported):
            Converted(nat, StateRequest(slices=slices), CircleEphemeris())
    with pytest.raises(DBEPUnsupported):                           # the entity used as a centre is itself re-centred
        Converted(TwoOsc(), StateRequest(slices=(SliceRequest("r", centre="AST"), SliceRequest("r:AST", centre="SSB"))), CircleEphemeris())


def test_no_request_is_identity():
    nat = eo.OscillatorBackend(differentiate=("k",))
    w = Converted(nat, None, CircleEphemeris())
    a, b = w.step(Z0, U0, 0.0, 10.0, P0), nat.step(Z0, U0, 0.0, 10.0, P0)
    assert np.max(np.abs(a.z - b.z)) < 1e-14 and w.describe().ephemeris == nat.describe().ephemeris


def test_orientation_service():
    svc = IAUOrientation({"MOON": dict(ra0=-1.3, dec0=1.1, W0=0.4, rate=2.66e-6)})
    errs = dbep.check_orientation(svc, "MOON", 1.2e8)
    assert set(errs) == {"t", "p.orient:MOON:ra0", "p.orient:MOON:dec0", "p.orient:MOON:W0", "p.orient:MOON:rate"}
    assert max(errs.values()) < 1e-6
    # the identity orientation: ra0 = -90 deg, dec0 = 90 deg, W = 0
    ident = IAUOrientation({"X": dict(ra0=-np.pi / 2, dec0=np.pi / 2, W0=0.0, rate=0.0)}, differentiated=("W0",))
    r = ident.orientation("X", 0.0)
    assert np.allclose(np.asarray(r.q), [1, 0, 0, 0]) and set(r.J["p"]) == {"orient:X:W0"}
    # body-fixed z axis of the Moon points to the pole (ra0, dec0)
    ra, dec = 0.7, 0.4
    pole = IAUOrientation({"B": dict(ra0=ra, dec0=dec, W0=1.0, rate=1e-5)}).orientation("B", 1e5)
    z_icrf = np.asarray(dbep.quat_rotate(pole.q, jnp.array([0.0, 0.0, 1.0])))
    assert np.allclose(z_icrf, [np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], atol=1e-12)
    # an estimated parameter changes the quaternion as the Jacobian says
    p2 = {"orient:B:ra0": ra + 1e-3}
    q0 = np.asarray(pole.q); q1 = np.asarray(IAUOrientation({"B": dict(ra0=ra, dec0=dec, W0=1.0, rate=1e-5)}).orientation("B", 1e5, p2).q)
    assert np.allclose(q1 - q0, 1e-3 * np.asarray(pole.J["p"]["orient:B:ra0"]), atol=1e-6)


def test_quaternion_algebra():
    q = dbep.quat_mul(dbep.quat_z(0.4), dbep.quat_x(-0.3))
    v = jnp.array([0.2, -0.7, 0.5])
    assert np.allclose(dbep.quat_rotate(dbep.quat_conj(q), dbep.quat_rotate(q, v)), v, atol=1e-14)
    assert np.allclose(dbep.quat_to_matrix(q) @ v, dbep.quat_rotate(q, v), atol=1e-14)


def test_requires_checks_controls_params_orientation():
    d = eo.OscillatorBackend(differentiate=("k",)).describe()
    dbep.check(Requires("orbital/1", controls=("dv",), params=("k", "g"), estimated=("k",)), d)
    with pytest.raises(DBEPMismatch) as e:
        dbep.check(Requires("orbital/1", controls=("acc",), params=("zz",), estimated=("g",), orientation=True), d)
    msg = str(e.value)
    for frag in ("no control 'acc'", "no parameter 'zz'", "parameter 'g' is not differentiated", "orientation required"):
        assert frag in msg


# ----------------------------------------------------------------------------- conformance-suite fixes (dbep 0.6.1)
T_2026 = 8.5e8          # TDB seconds past J2000 in 2026: where real back-ends run


class _Edit:
    """Wraps a back-end and edits its results, to show that a check catches a wrong implementation."""

    def __init__(self, native, edit):
        self._n, self._edit = native, edit

    def describe(self):
        return self._n.describe()

    def step(self, *a):
        return self._edit(self._n.step(*a))


def _with_J(res, **blocks):
    return StepResult(res.z, {**res.J, **blocks}, res.status, res.meta, res._dense)


def test_step_jacobians_at_a_real_epoch():
    """The epoch is perturbed by an absolute step: with a step relative to |t| (850 s at 2026) J["t"] failed by 1e-3."""
    b = eo.OscillatorBackend(differentiate=("k",))
    errs = dbep.check_step_jacobians(b, Z0, U0, T_2026, 10.0, P0)
    assert errs["t"] < 1e-6 and errs["dt"] < 1e-6
    wrong = _Edit(b, lambda r: _with_J(r, t=r.J["t"] * 0.0))        # a back-end that forgets the explicit time dependence
    with pytest.raises(AssertionError, match="Jacobian block t"):
        dbep.check_step_jacobians(wrong, Z0, U0, T_2026, 10.0, P0)


class _Km:
    """The toy with its state scaled by 1e5, like positions in km: every block grows by 1e5 except J["z"]."""
    S = 1e5

    def __init__(self):
        self._n = eo.OscillatorBackend(differentiate=("k",))

    def describe(self):
        return self._n.describe()

    def step(self, z, u, t, dt, p):
        r = self._n.step(jnp.asarray(z) / self.S, {k: jnp.asarray(v) / self.S for k, v in u.items()}, t, dt, p)
        J = {"z": r.J["z"], "t": self.S * r.J["t"], "dt": self.S * r.J["dt"], "u": r.J["u"],
             "p": {k: self.S * v for k, v in r.J["p"].items()}}
        return StepResult(self.S * r.z, J, r.status, r.meta, None)


def test_tolerance_is_relative_to_the_block():
    """Blocks of size 1e5 pass with the relative error, and an error of 1e-4 relative is still caught."""
    km = _Km()
    z, u = _Km.S * Z0, {"dv": _Km.S * U0["dv"]}
    errs = dbep.check_step_jacobians(km, z, u, 0.0, 10.0, P0)
    assert max(errs.values()) < 1e-6
    wrong = _Edit(km, lambda r: _with_J(r, p={"k": r.J["p"]["k"] * (1 + 1e-4)}))
    with pytest.raises(AssertionError, match="Jacobian block p.k"):
        dbep.check_step_jacobians(wrong, z, u, 0.0, 10.0, P0)
    # per-block settings: a dict with a default
    dbep.check_step_jacobians(km, z, u, 0.0, 10.0, P0, tol={"default": 1e-6, "p.k": 1e-3})


def test_check_refuses_other_protocol_versions():
    d = eo.OscillatorBackend(differentiate=("k",)).describe()
    dbep.check(Requires("orbital/1"), d)
    for bad in ("DBE-P/1.0", "DBE-P/7.3", "DBE-P/0.5", "DBE-P/0.7", "XYZ/0.6", "DBE-P", "garbage"):
        with pytest.raises(DBEPMismatch, match="protocol"):
            dbep.check(Requires("orbital/1"), dataclasses.replace(d, protocol=bad))


def test_results_cross_jit():
    b = eo.OscillatorBackend(differentiate=("k",))
    plain = b.step(Z0, U0, 0.0, 10.0, P0)
    res = jax.jit(lambda z: b.step(z, U0, 0.0, 10.0, P0))(Z0)
    assert isinstance(res, StepResult) and res.meta == plain.meta
    assert np.max(np.abs(np.asarray(res.z) - np.asarray(plain.z))) < 1e-12
    assert np.max(np.abs(np.asarray(res.J["p"]["k"]) - np.asarray(plain.J["p"]["k"]))) < 1e-12
    with pytest.raises(RuntimeError, match="inside the function"):
        res.dense(np.array([5.0]))
    dn = jax.jit(lambda z: b.step(z, U0, 0.0, 10.0, P0).dense(jnp.array([5.0, 10.0]), partials=True))(Z0)
    assert dn.meta == plain.meta and np.asarray(dn.J["z"]).shape == (2, 6, 6)
    w = Converted(b, WANT_SSB, CircleEphemeris())                  # the wrapper too
    assert jax.jit(lambda z: w.step(z, U0, 0.0, 10.0, P0))(Z0).meta == w.describe().state_meta()
    assert isinstance(hash(b.describe()), int)                      # a descriptor can be a static argument


def test_check_meta_catches_wrong_row_count():
    b = eo.OscillatorBackend(differentiate=("k",))
    wrong = _Edit(b, lambda r: _with_J(r, z=r.J["z"][:5]))          # (5, 6): columns right, rows wrong
    with pytest.raises(AssertionError):
        dbep.check_meta(wrong, Z0, U0, 0.0, 10.0, P0)


def test_chaining_composes_every_block():
    b = eo.OscillatorBackend(differentiate=("k",))
    errs = dbep.check_chaining(b, Z0, U0, 0.0, 10.0, 20.0, P0)
    assert set(errs) == {"state", "stm", "t", "dt", "p.k"}
    # a constant offset in J["p"] passes a single-step comparison of z but breaks S_k = Phi_k S_k-1 + J_k["p"]
    wrong = _Edit(b, lambda r: _with_J(r, p={"k": r.J["p"]["k"] + 1e-3}))
    with pytest.raises(AssertionError, match="p.k"):
        dbep.check_chaining(wrong, Z0, U0, 0.0, 10.0, 20.0, P0)
    acc = _Edit(b, lambda r: r)
    acc.describe = lambda: dataclasses.replace(b.describe(), controls=b.describe().controls + (Control("acc", (2, 3), "acceleration", "km/s^2"),))
    with pytest.raises(ValueError, match="only impulses"):
        dbep.check_chaining(acc, Z0, {**U0, "acc": np.ones((2, 3))}, 0.0, 10.0, 20.0, P0)


def test_centre_refused_on_a_slice_without_one():
    class WithGm:
        def describe(self):
            d = eo.OscillatorBackend().describe()
            return dataclasses.replace(d, state=d.state + (Slice("gm:X", 1, "gm", "km^3/s^2"),))
    with pytest.raises(DBEPUnsupported, match="role 'gm'"):
        Converted(WithGm(), StateRequest((SliceRequest("gm:X", centre="SUN"),)), CircleEphemeris())


class _BigParam:
    """The toy with its parameter k expressed as K = k * 1e10: J["p"]["K"] is 1e-10 times the toy's, like the GM of the Sun in an orbit model."""
    S = 1e10

    def __init__(self):
        self._n = eo.OscillatorBackend(differentiate=("k",))
        d = self._n.describe()
        self._d = dataclasses.replace(d, params=(Param("K", (), 1.0 * self.S, "1/s^2 * 1e10", True), Param("g", (), 0.0, "km/s^2", False)))
        self._meta = self._d.state_meta()

    def describe(self):
        return self._d

    def step(self, z, u, t, dt, p):
        r = self._n.step(z, u, t, dt, {"k": jnp.asarray(p.get("K", self.S)) / self.S, "g": p.get("g", 0.0)})
        J = {**r.J, "p": {"K": r.J["p"]["k"] / self.S}}
        return StepResult(r.z, J, r.status, self._meta, None)


def test_small_block_of_a_large_parameter_is_checked_relative_to_its_effect():
    """J["p"]["K"] is about 1e-10: divided by max(1, |J|) its error was absolute, so a factor of 2 passed. Scaled by |K| it is caught."""
    b = _BigParam()
    p = {"K": 0.8 * _BigParam.S, "g": 0.05}
    assert dbep.check_step_jacobians(b, Z0, U0, 0.0, 10.0, p)["p.K"] < 1e-6
    wrong = _Edit(b, lambda r: _with_J(r, p={"K": 2.0 * r.J["p"]["K"]}))
    with pytest.raises(AssertionError, match="Jacobian block p.K"):
        dbep.check_step_jacobians(wrong, Z0, U0, 0.0, 10.0, p)
