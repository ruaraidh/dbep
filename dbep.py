"""DBE-P: dynamics back-end protocol, draft 0.7.

Types, a requirement check and a conformance suite. No dynamics live here.
A *back-end* implements `Backend`; a *consumer* states what it needs with
`Requires`, and `check()` compares that with the back-end's `Descriptor`.

Conventions (core):
  * `step` maps (z, u, t, dt, p) -> z' at t + dt.
  * Inputs are a tree {"z": array, "u": {name: array}, "t": scalar, "dt": scalar, "p": {name: array}}.
  * There is no per-call selection of derivatives. A back-end returns ALL of them:
    "z", every control in "u", every parameter in "p" whose descriptor entry has `differentiated=True`,
    "t" and "dt". What is differentiated is fixed when the model is built.
  * The Jacobian tree `J` mirrors the inputs: a leaf of shape s gives J[...] of shape (n_z,) + s.
  * The consumer states what z must be (`StateRequest`: centre, axes, unit per slice) when the model is built.
    The back-end MUST return exactly that, converting internally if its native coordinates differ, with
    Jacobians of the requested state, or refuse at construction (`DBEPUnsupported`). There is no outputs service.
  * Every result carries `meta` (StateMeta): what z IS (layout, centre, axes, units, time, ephemeris id)
    and what the Jacobian tree contains, so results are self-describing.
  * `dense(t_vec, partials=False)` returns states at epochs inside [t, t + dt]; with `partials=True` it also
    returns the Jacobian tree at those epochs, relative to the START of the step. Its "dt" block is the
    sensitivity to the evaluation offset (the observation time tag).
  * Inside jit all leaves are JAX arrays and dict keys are static strings; `meta` is static.
"""
from __future__ import annotations

import builtins
import dataclasses
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)          # the protocol is float64

Array = Any
PROTOCOL = "DBE-P/0.7"


# ----------------------------------------------------------------------------- descriptor pieces
@dataclass(frozen=True)
class TimeConvention:
    scale: str = "TDB"
    epoch: str = "J2000"
    unit: str = "s"


@dataclass(frozen=True)
class Slice:
    name: str            # "r", "v", "gm:EROS", "r:EROS", ...
    size: int
    role: str            # "position", "velocity", "gm", "mass", ...
    unit: str = ""
    axes: str = ""       # orientation of the axes the components refer to, e.g. "ICRF"
    centre: str = ""     # origin the quantity is measured from, e.g. "EROS"; the profile defines the id space


@dataclass(frozen=True)
class SliceRequest:
    """What the consumer wants one slice of z to be. Empty fields mean "back-end's choice"."""
    name: str
    centre: str = ""
    axes: str = ""
    unit: str = ""


@dataclass(frozen=True)
class StateRequest:
    """Part of the model specification: the coordinates the consumer expects in z. The back-end MUST honour
    it (converting if needed) or refuse at construction."""
    slices: tuple = ()               # tuple[SliceRequest]

    def for_slice(self, name: str):
        for r in self.slices:
            if r.name == name:
                return r
        return None


class DBEPUnsupported(Exception):
    """The back-end cannot provide the requested coordinates."""


@dataclass(frozen=True)
class Control:
    name: str            # "dv", "acc", ...
    shape: tuple
    role: str            # "delta_v", "acceleration", "thrust", ...
    unit: str = ""
    frame: str = "ICRF"  # axes of the control's vector components (section 9): "ICRF", "RTN:<CENTRE>" (radial, transverse, normal) or
                         # "VNC:<CENTRE>" (velocity, normal, co-normal), both of the state relative to CENTRE before the control acts.
                         # Named with its centre, so it stays defined after a conversion.

    def __post_init__(self):
        if not valid_frame(self.frame):
            raise ValueError(f"control {self.name!r}: frame {self.frame!r} is not 'ICRF', 'RTN:<CENTRE>' or 'VNC:<CENTRE>'")


def valid_frame(frame: str) -> bool:
    """'ICRF', or 'RTN:<CENTRE>' / 'VNC:<CENTRE>' with a centre name that follows the naming rule (no '.', ',' or ':')."""
    if frame == "ICRF":
        return True
    kind, _, centre = frame.partition(":")
    return kind in ("RTN", "VNC") and bool(centre) and not any(ch in centre for ch in ".,:")


@dataclass(frozen=True)
class Param:
    name: str            # "k", "cr_a_m", "gm:EROS", ...
    shape: tuple = ()
    default: Any = 0.0
    unit: str = ""
    differentiated: bool = True      # True: appears in J["p"]; False: held fixed at the value passed in p


@dataclass(frozen=True)
class Capabilities:
    level: int                       # 0: step, J["z"]  1: +dense  2: +u/p/t/dt blocks
    dense: bool = False
    batch: bool = False
    jit: bool = True
    reverse: bool = False            # reverse-mode differentiable?
    orientation: bool = False        # provides orientation(body, t, p) (section 8.6)


@dataclass(frozen=True)
class StateMeta:
    """What the state vector z is, and what the Jacobian tree holds. Static; attached to every result."""
    protocol: str
    profile: str
    time: TimeConvention
    slices: tuple                    # tuple[Slice], in the order they appear in z
    ephemeris: str                   # id of the ephemeris/constants set the model uses ("" if none)
    jacobian: tuple                  # tuple[(dotted path, shape)] of every leaf of the Jacobian tree

    @property
    def n_z(self) -> int:
        return sum(s.size for s in self.slices)

    def index(self, name: str) -> builtins.slice:
        i = 0
        for s in self.slices:
            if s.name == name:
                return builtins.slice(i, i + s.size)
            i += s.size
        raise KeyError(name)

    def slices_with_role(self, role: str) -> list:
        return [s for s in self.slices if s.role == role]


@dataclass(frozen=True)
class Descriptor:
    profile: str                     # e.g. "orbital/1"
    time: TimeConvention
    state: tuple                     # tuple[Slice]
    controls: tuple                  # tuple[Control]
    params: tuple                    # tuple[Param]
    capabilities: Capabilities
    ephemeris: str = ""
    status_bits: dict = field(default_factory=lambda: {1: "max_steps", 2: "underflow", 3: "out_of_coverage",
                                                        4: "impact", 5: "non_finite"},
                              hash=False)    # compared, but left out of the hash so a Descriptor can be a static jit argument
    protocol: str = PROTOCOL

    @property
    def n_z(self) -> int:
        return sum(s.size for s in self.state)

    def roles(self) -> set:
        return {s.role for s in self.state}

    def differentiated_params(self) -> tuple:
        return tuple(q.name for q in self.params if q.differentiated)

    def state_meta(self) -> StateMeta:
        leaves_ = [("z", (self.n_z,)), ("t", ()), ("dt", ())]
        leaves_ += [(f"u.{c.name}", tuple(c.shape)) for c in self.controls]
        leaves_ += [(f"p.{q.name}", tuple(q.shape)) for q in self.params if q.differentiated]
        return StateMeta(self.protocol, self.profile, self.time, tuple(self.state), self.ephemeris,
                         tuple(leaves_))


# ----------------------------------------------------------------------------- results
@dataclass
class StepResult:
    z: Array                                         # (n_z,) at t + dt
    J: dict                                          # mirrors the inputs, see module docstring
    status: Array                                    # int bitmask
    meta: StateMeta                                  # what z is
    _dense: Optional[Callable] = None

    def dense(self, t_vec: Array, partials: bool = False) -> "DenseResult":
        """States at epochs t_vec inside [t, t+dt]; with partials=True also the Jacobian tree.
        `partials` is a static Python bool (changing it may recompile). Without it a back-end
        SHOULD skip the variational work for the interpolation, which is the cheap path."""
        if self._dense is None:
            raise NotImplementedError("back-end does not provide dense output (level < 1)")
        return self._dense(t_vec, partials)


@dataclass
class DenseResult:
    z: Array                                         # (n_t, n_z)
    meta: StateMeta
    J: Optional[dict] = None                         # same tree as StepResult.J with a leading n_t axis; None if partials=False


def _dense_not_carried(t_vec, partials=False):
    raise RuntimeError("this StepResult was rebuilt from its leaves (for example returned from jax.jit), which does not carry "
                       "dense output: call res.dense(...) inside the function that called step")


# Results are pytrees, so they can be returned from jax.jit, vmapped and tree-mapped. The arrays are the leaves; `meta` is static
# (aux data). The dense closure is not an array and may hold tracers, so it is not carried across a rebuild.
jax.tree_util.register_pytree_node(
    StepResult,
    lambda r: ((r.z, r.J, r.status), r.meta),
    lambda meta, ch: StepResult(z=ch[0], J=ch[1], status=ch[2], meta=meta, _dense=_dense_not_carried))
jax.tree_util.register_pytree_node(
    DenseResult,
    lambda r: ((r.z, r.J), r.meta),
    lambda meta, ch: DenseResult(z=ch[0], meta=meta, J=ch[1]))


class Backend(Protocol):
    def describe(self) -> Descriptor: ...

    def step(self, z: Array, u: dict, t: Array, dt: Array, p: dict) -> StepResult: ...


# ----------------------------------------------------------------------------- consumer side
@dataclass(frozen=True)
class Requires:
    profile: str
    level: int = 0
    roles: tuple = ()                                # state-slice roles the consumer needs
    state: Optional[StateRequest] = None             # coordinates the consumer expects in z
    dense: bool = False
    controls: tuple = ()                             # control names the model must declare
    params: tuple = ()                               # parameter names the model must declare
    estimated: tuple = ()                            # parameter names that must be differentiated (appear in J["p"])
    orientation: bool = False                        # the back-end must provide orientation()
    control_frames: tuple = ()                       # ((control name, frame), ...): the frame each named control must be given in


class DBEPMismatch(Exception):
    pass


def _version(protocol: str):
    """'DBE-P/0.6' -> ('DBE-P', 0, 6); None if the string is not of that form."""
    name, _, ver = protocol.partition("/")
    major, _, minor = ver.partition(".")
    try:
        return name, int(major), int(minor)
    except ValueError:
        return None


def compatible_protocol(protocol: str) -> bool:
    """Section 16: the name and the major version must match. While the protocol is a draft (major 0) any change may break,
    so the minor version must match as well."""
    have, want = _version(protocol), _version(PROTOCOL)
    if have is None or have[:2] != want[:2]:
        return False
    return want[1] != 0 or have[2] == want[2]


def check(req: Requires, desc: Descriptor) -> None:
    """Raise with every unmet requirement named, or return None."""
    problems = []
    if not compatible_protocol(desc.protocol):
        problems.append(f"protocol {desc.protocol!r} is not compatible with {PROTOCOL!r} (section 16)")
    if desc.profile != req.profile:
        problems.append(f"profile {desc.profile!r} != required {req.profile!r}")
    if desc.capabilities.level < req.level:
        problems.append(f"capability level {desc.capabilities.level} < required {req.level}")
    if req.dense and not desc.capabilities.dense:
        problems.append("dense output required but not provided")
    for r in req.roles:
        if r not in desc.roles():
            problems.append(f"no state slice with role {r!r}")
    if req.orientation and not desc.capabilities.orientation:
        problems.append("orientation required but not provided")
    have_c = {c.name: c for c in desc.controls}
    for c in req.controls:
        if c not in have_c:
            problems.append(f"no control {c!r}")
    for c, frame in req.control_frames:
        if c not in have_c:
            problems.append(f"no control {c!r} (frame {frame!r} required)")
        elif have_c[c].frame != frame:
            problems.append(f"control {c!r}: frame is {have_c[c].frame!r}, required {frame!r}")
    have_p = {q.name: q for q in desc.params}
    for q in req.params:
        if q not in have_p:
            problems.append(f"no parameter {q!r}")
    for q in req.estimated:
        if q not in have_p:
            problems.append(f"no parameter {q!r} to estimate")
        elif not have_p[q].differentiated:
            problems.append(f"parameter {q!r} is not differentiated")
    if req.state is not None:
        for sr in req.state.slices:
            have = next((s for s in desc.state if s.name == sr.name), None)
            if have is None:
                problems.append(f"requested slice {sr.name!r} is not in the state")
                continue
            for attr in ("centre", "axes", "unit"):
                want = getattr(sr, attr)
                if want and getattr(have, attr) != want:
                    problems.append(f"slice {sr.name!r}: {attr} is {getattr(have, attr)!r}, requested {want!r}")
    if problems:
        raise DBEPMismatch("; ".join(problems))


# ----------------------------------------------------------------------------- conversion wrapper
class EphemerisProvider(Protocol):
    """What a conversion needs from an ephemeris: the state of a body relative to the SSB."""
    id: str

    def state(self, body: str, t: Array) -> Array:
        """(6,) position [km] and velocity [km/s] of `body` relative to the SSB in ICRF axes at TDB epoch `t` [s].
        MUST be traceable and differentiable in `t` (the wrapper differentiates it for the time terms). Raise
        KeyError for a body it does not have."""


class Converted:
    """A back-end honouring a StateRequest, built from a native back-end (DBE-P section 8.2).

    Position/velocity pairs (`r`/`v`, `r:X`/`v:X`) whose native centre differs from the requested one are translated:
    F_req(z, t, dt) = T(t+dt)(F(T(t)^-1(z), t, dt)), with T(t)(z) = z + m(z, t) and m the state of the native centre
    relative to the requested one. A centre may be the SSB, a body of the ephemeris, or an entity in z (slices `r:X`,
    `v:X` that are not themselves re-centred). Every Jacobian block is that of the requested state, including the
    time terms of a moving centre. Axes are ICRF only and units are not converted: other requests are refused."""

    def __init__(self, native, request: Optional[StateRequest], ephemeris: EphemerisProvider):
        self._native, self._eph = native, ephemeris
        nd = native.describe()
        off, idx = 0, {}
        for s in nd.state:
            idx[s.name] = (builtins.slice(off, off + s.size), s)
            off += s.size
        self._n = off
        for sr in (request.slices if request else ()):
            if sr.name not in idx:
                raise DBEPUnsupported(f"requested slice {sr.name!r} is not in the native state")
            have = idx[sr.name][1]
            if sr.axes and sr.axes != "ICRF":
                raise DBEPUnsupported(f"axes {sr.axes!r} not supported (only ICRF)")
            if sr.axes and have.axes and have.axes != sr.axes:
                raise DBEPUnsupported(f"slice {sr.name!r}: native axes {have.axes!r} cannot be converted to {sr.axes!r}")
            if sr.unit and sr.unit != have.unit:
                raise DBEPUnsupported(f"slice {sr.name!r}: unit {sr.unit!r} requested, native is {have.unit!r}")
            if sr.centre and have.role not in ("position", "velocity"):
                raise DBEPUnsupported(f"slice {sr.name!r} has role {have.role!r}: a centre can only be requested for a position or velocity")
        self._entities = {}                                   # body -> (r slice, v slice, native centre)
        for name, (sl, s) in idx.items():
            if s.role == "position" and name.startswith("r:") and "v:" + name[2:] in idx:
                self._entities[name[2:]] = (sl, idx["v:" + name[2:]][0], s.centre)
        conv, new = [], {name: s for name, (sl, s) in idx.items()}
        for name, (sl, s) in idx.items():
            if s.role != "position" or not (name == "r" or name.startswith("r:")):
                continue
            vname = "v" + name[1:]
            if vname not in idx:
                continue
            rq, vq = (request.for_slice(name), request.for_slice(vname)) if request else (None, None)
            cs = {x.centre for x in (rq, vq) if x is not None and x.centre}
            if len(cs) > 1:
                raise DBEPUnsupported(f"{name!r} and {vname!r} must be requested relative to one centre")
            cr = cs.pop() if cs else s.centre
            if cr != s.centre:
                conv.append((sl, idx[vname][0], s.centre, cr, name))
                new[name] = dataclasses.replace(s, centre=cr)
                new[vname] = dataclasses.replace(idx[vname][1], centre=cr)
        for _, _, _, cr, name in conv:
            e = cr if cr in self._entities else None
            if e is not None and ("r:" + e) in [c[4] for c in conv]:
                raise DBEPUnsupported(f"centre {cr!r} is an entity in z that is itself re-centred in this request")
        self._conv = conv
        # When no centre of a conversion is an entity of z, T(t)(z) = z + b(t): its state Jacobian is the identity, and the native state is z - b(t).
        # The general path (a Jacobian, a solve, an inverse) is then exact but needless work at every step and dense epoch.
        self._translation = not any(c in self._entities for _, _, cn, cr, _ in conv for c in (cn, cr))
        for _, _, cn, cr, _ in conv:                          # every centre must be resolvable (and acyclic)
            for c in (cn, cr):
                try:
                    self._X(c, jnp.zeros(self._n), 0.0)
                except KeyError:
                    raise DBEPUnsupported(f"no ephemeris for centre {c!r}") from None
        eph = nd.ephemeris
        if conv and ephemeris.id not in eph.split("+"):
            eph = f"{eph}+{ephemeris.id}" if eph else ephemeris.id
        self._d = dataclasses.replace(nd, state=tuple(new[s.name] for s in nd.state), ephemeris=eph)
        self._meta = self._d.state_meta()

    # -- conversion pieces
    def _X(self, name, z, t, depth=0):
        """State [r; v] of a centre relative to the SSB, taken from the entity slices of z or from the ephemeris."""
        if name == "SSB":
            return jnp.zeros(6)
        if name in self._entities:
            if depth > 8:
                raise DBEPUnsupported("cycle in the centres of integrated bodies")
            ri, vi, centre = self._entities[name]
            return jnp.concatenate([z[ri], z[vi]]) + self._X(centre, z, t, depth + 1)
        return jnp.asarray(self._eph.state(name, t), dtype=jnp.float64)

    def _T(self, z, t):
        out = z
        for ri, vi, cn, cr, _ in self._conv:
            m = self._X(cn, z, t) - self._X(cr, z, t)
            out = out.at[ri].add(m[:3]).at[vi].add(m[3:])
        return out

    def _start(self, z, t):
        z0 = jnp.zeros(self._n)
        b = self._T(z0, t)
        if self._translation:
            zn = z - b
            return zn, jnp.eye(self._n), jax.jacfwd(self._T, 1)(zn, t)
        M = jax.jacfwd(self._T, 0)(z0, t)                     # T is affine in z
        zn = jnp.linalg.solve(M, z - b)
        return zn, jnp.linalg.inv(M), jax.jacfwd(self._T, 1)(zn, t)

    def _end(self, zn, te):
        Me = jnp.eye(self._n) if self._translation else jax.jacfwd(self._T, 0)(zn, te)
        return self._T(zn, te), Me, jax.jacfwd(self._T, 1)(zn, te)

    @staticmethod
    def _jtree(Jn, Me, dTe, M0inv, dT0):
        out = {}
        for k, v in Jn.items():
            if k == "z":
                out["z"] = Me @ v @ M0inv
            elif k == "t":
                out["t"] = Me @ (v - Jn["z"] @ M0inv @ dT0) + dTe
            elif k == "dt":
                out["dt"] = Me @ v + dTe
            else:                                             # "u" and "p": a dict of blocks (n_z,) + s
                out[k] = {n: jnp.tensordot(Me, a, axes=1) for n, a in v.items()}
        return out

    # -- protocol
    def describe(self) -> Descriptor:
        return self._d

    def step(self, z, u, t, dt, p):
        z = jnp.asarray(z, dtype=jnp.float64); t = jnp.asarray(t, dtype=jnp.float64); dt = jnp.asarray(dt, dtype=jnp.float64)
        zn, M0inv, dT0 = self._start(z, t)
        res = self._native.step(zn, u, t, dt, p)
        zend, Me, dTe = self._end(jnp.asarray(res.z), t + dt)
        J = self._jtree(res.J, Me, dTe, M0inv, dT0)

        def dense(t_vec, partials=False):
            dn = res.dense(t_vec, partials=partials)
            tv = jnp.asarray(t_vec, dtype=jnp.float64)
            zi, Mi, dTi = jax.vmap(self._end)(jnp.asarray(dn.z), tv)
            if not partials:
                return DenseResult(z=zi, meta=self._meta, J=None)
            Ji = jax.vmap(self._jtree, in_axes=(0, 0, 0, None, None))(dn.J, Mi, dTi, M0inv, dT0)
            return DenseResult(z=zi, meta=self._meta, J=Ji)

        return StepResult(z=zend, J=J, status=res.status, meta=self._meta, _dense=dense)


# ----------------------------------------------------------------------------- orientation (section 8.6)
def quat_mul(a, b):
    """Hamilton product, quaternions scalar-first (w, x, y, z)."""
    aw, ax, ay, az = a[0], a[1], a[2], a[3]
    bw, bx, by, bz = b[0], b[1], b[2], b[3]
    return jnp.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                      aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw])


def quat_conj(q):
    return jnp.stack([q[0], -q[1], -q[2], -q[3]])


def quat_rotate(q, v):
    """Rotate the 3-vector v by the unit quaternion q: v' = q v q*. For an orientation quaternion this maps body-fixed
    components to ICRF components."""
    return quat_mul(quat_mul(q, jnp.concatenate([jnp.zeros(1), v])), quat_conj(q))[1:]


def quat_to_matrix(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    return jnp.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def quat_z(a):
    return jnp.stack([jnp.cos(a / 2), 0.0 * a, 0.0 * a, jnp.sin(a / 2)])


def quat_x(a):
    return jnp.stack([jnp.cos(a / 2), jnp.sin(a / 2), 0.0 * a, 0.0 * a])


def iau_quaternion(ra0, dec0, W0, rate, t):
    """IAU rotation model with linear terms: R = Rz(ra0 + 90 deg) Rx(90 deg - dec0) Rz(W), W = W0 + rate * t,
    as a unit quaternion taking body-fixed components to ICRF components. Angles in rad, rate in rad/s, t in TDB s past J2000."""
    return quat_mul(quat_mul(quat_z(ra0 + jnp.pi / 2), quat_x(jnp.pi / 2 - dec0)), quat_z(W0 + rate * t))


@dataclass
class OrientationResult:
    """q: (4,) unit quaternion, scalar-first, rotating body-fixed components to ICRF components. J mirrors the inputs
    like the step's J: J["t"] is dq/dt (4,), J["p"][name] is dq/dparameter, shape (4,) + s, for every differentiated
    orientation parameter. The sign of q is not canonical but q is continuous in t and in the parameters."""
    q: Array
    J: dict
    body: str = ""


class IAUOrientation:
    """Reference orientation service: the IAU model with linear terms (periodic terms later). Pure JAX. It is the shared
    implementation a back-end and a consumer can both use, so that they agree."""
    SUBS = ("ra0", "dec0", "W0", "rate")
    UNITS = {"ra0": "rad", "dec0": "rad", "W0": "rad", "rate": "rad/s"}

    def __init__(self, bodies: dict, differentiated=SUBS, id: str = "iau-linear"):
        self.bodies = {b: {k: float(v) for k, v in d.items()} for b, d in bodies.items()}
        self._diff = tuple(differentiated)
        self.id = id

    def params(self) -> tuple:
        return tuple(Param(f"orient:{b}:{s}", (), self.bodies[b][s], self.UNITS[s], s in self._diff)
                     for b in self.bodies for s in self.SUBS)

    def orientation(self, body: str, t, p: Optional[dict] = None) -> OrientationResult:
        p = p or {}
        vals = {s: jnp.asarray(p.get(f"orient:{body}:{s}", self.bodies[body][s]), dtype=jnp.float64) for s in self.SUBS}
        t = jnp.asarray(t, dtype=jnp.float64)

        def q_of(tt, v):
            return iau_quaternion(v["ra0"], v["dec0"], v["W0"], v["rate"], tt)

        Jt = jax.jacfwd(q_of, 0)(t, vals)
        Jv = jax.jacfwd(q_of, 1)(t, vals)
        Jp = {f"orient:{body}:{s}": Jv[s] for s in self.SUBS if s in self._diff}
        return OrientationResult(q=q_of(t, vals), J={"t": Jt, "p": Jp}, body=body)


def check_orientation(service, body, t, p=None, h=1e-6, tol=1e-6) -> dict:
    """Orientation conformance: unit quaternion, proper rotation matrix, and every Jacobian leaf against central differences
    (steps scaled so that q moves by about h; errors relative to max(1, |dq/dx|))."""
    res = service.orientation(body, t, p)
    q = np.asarray(res.q, float)
    assert abs(np.linalg.norm(q) - 1.0) < 1e-12, "quaternion is not unit"
    R = np.asarray(quat_to_matrix(res.q))
    assert np.max(np.abs(R @ R.T - np.eye(3))) < 1e-12 and abs(np.linalg.det(R) - 1.0) < 1e-12, "not a proper rotation"
    v = np.array([0.3, -0.5, 0.8])
    assert np.max(np.abs(np.asarray(quat_rotate(res.q, v)) - R @ v)) < 1e-12, "quat_rotate and matrix disagree"
    errs = {}
    Jt = np.asarray(res.J["t"])
    ht = h / max(1.0, float(np.linalg.norm(Jt)))
    fd = (np.asarray(service.orientation(body, t + ht, p).q) - np.asarray(service.orientation(body, t - ht, p).q)) / (2 * ht)
    errs["t"] = float(np.max(np.abs(fd - Jt))) / max(1.0, float(np.linalg.norm(Jt)))
    defaults = {q_.name: q_.default for q_ in service.params()}
    base = dict(p or {})
    for name, Jb in res.J["p"].items():
        Jb = np.asarray(Jb)
        x0 = float(base.get(name, defaults[name]))
        scale = max(1.0, float(np.linalg.norm(Jb)))
        hp = h / scale                                    # a step that moves q by about h
        qp = np.asarray(service.orientation(body, t, {**base, name: x0 + hp}).q)
        qm = np.asarray(service.orientation(body, t, {**base, name: x0 - hp}).q)
        errs["p." + name] = float(np.max(np.abs((qp - qm) / (2 * hp) - Jb))) / scale      # relative to the derivative's size
    for k, e in errs.items():
        assert e < tol, f"orientation Jacobian {k}: relative error {e:.2e} > {tol:.0e}"
    return errs


# ----------------------------------------------------------------------------- tree helpers
def leaves(tree, path=()):
    if isinstance(tree, dict):
        for k, v in tree.items():
            yield from leaves(v, path + (k,))
    else:
        yield path, tree


def get(tree, path):
    for k in path:
        tree = tree[k]
    return tree


def put(tree, path, value):
    if not path:
        return value
    out = dict(tree)
    out[path[0]] = put(tree[path[0]], path[1:], value)
    return out


# ----------------------------------------------------------------------------- conformance suite
def zero_controls(desc: Descriptor) -> dict:
    return {c.name: np.zeros(c.shape) for c in desc.controls}


def complete(desc: Descriptor, u: dict, p: dict) -> tuple:
    """Fill missing controls with zeros and missing parameters with their defaults."""
    uu = {**zero_controls(desc), **u}
    pp = {q.name: np.asarray(q.default, float) * np.ones(q.shape) for q in desc.params}
    pp.update({k: np.asarray(v, float) for k, v in p.items()})
    return uu, pp


def check_meta(backend: Backend, z, u, t, dt, p) -> None:
    """Result metadata matches the descriptor and the actual arrays."""
    d = backend.describe()
    res = backend.step(z, u, t, dt, p)
    assert res.meta == d.state_meta(), "result meta differs from the descriptor's"
    n_z = res.meta.n_z
    assert np.asarray(res.z).shape == (n_z,), "z does not have the length the meta declares"
    got = {(".".join(path), tuple(np.asarray(leaf).shape[1:])) for path, leaf in leaves(res.J)}
    assert got == set(res.meta.jacobian), f"Jacobian tree differs from meta: {got ^ set(res.meta.jacobian)}"
    for path, leaf in leaves(res.J):
        assert np.asarray(leaf).shape[:1] == (n_z,), f"Jacobian block {'.'.join(path)} has {np.asarray(leaf).shape[:1]} rows, not n_z = {n_z}"
    if d.capabilities.dense:
        dn = res.dense(np.array([t + dt]), partials=True)
        assert dn.meta == res.meta, "dense meta differs from step meta"


def _per_leaf(value, path: str):
    """A setting given either as one number or as a dict {dotted path: number} (paths not in the dict take its "default", if any)."""
    if isinstance(value, dict):
        return value.get(path, value.get("default"))
    return value


TIME_LEAVES = ("t", "dt")


def _input_size(x, meta=None):
    """The size each component's step is scaled by: max(1, max |x|) over the quantity the component belongs to (one state slice of z,
    or the whole input array), not the component alone, so that a near-zero component of a large vector is not given a tiny step."""
    x = np.abs(np.asarray(x, float))
    if meta is None:
        return np.full(x.shape, max(1.0, float(np.max(x))) if x.size else 1.0)
    out = np.empty(x.shape)
    for s in meta.slices:
        sl = meta.index(s.name)
        out[sl] = max(1.0, float(np.max(x[sl])))
    return out


def check_step_jacobians(backend: Backend, z, u, t, dt, p, h=1e-6, tol=1e-6, h_time=1e-3) -> dict:
    """Every leaf of the returned Jacobian tree against central differences of `step`.

    Steps: a component of an input is perturbed by h * max(1, max |x|), the maximum taken over its state slice (for z) or over the
    whole input array (for a control or parameter), except the epoch inputs `t` and `dt`, which are perturbed by the absolute
    step `h_time` [s]: an epoch is an offset from J2000, so its size says nothing about the time scale of the dynamics (at a 2026
    epoch a relative step would be about 850 s). The difference quotient divides by the step actually taken, (x+h) - (x-h) in
    float64, which matters for epochs near 1e9 s where float64 resolves only about 1e-7 s.

    Errors are scale-free: each column is multiplied by the size s of its input (the max(1, max |x|) of the steps; 1 s for `t` and `dt`),
    so that J s is the state change for a change of the input of its own size, and the error of the block is max |FD - J| s divided by
    max(1, max |J s|). A parameter with a large value and a small derivative (the GM of the Sun: J of order 1e-11 km per km^3/s^2, GM
    1.3e11) is then checked relative to its effect, not absolutely; dividing by max(1, max |J|) alone let a block that was wrong by a
    factor of two pass.

    `h` and `tol` may be dicts {dotted path: value}, with an optional "default", so a back-end can state its tolerance per block.
    Returns the error of every leaf."""
    d = backend.describe()
    u, p = complete(d, u, p)
    inp = {"z": z, "u": u, "t": t, "dt": dt, "p": p}
    res = backend.step(z, u, t, dt, p)

    def run(i):
        return np.asarray(backend.step(i["z"], i["u"], i["t"], i["dt"], i["p"]).z, float)

    errs = {}
    for path, Jb in leaves(res.J):
        name = ".".join(path)
        x = np.asarray(get(inp, path), float)
        Jb = np.asarray(Jb, float)
        h_leaf = _per_leaf(h, name)
        size = np.ones(x.shape) if name in TIME_LEAVES else _input_size(x, res.meta if name == "z" else None)
        worst = scale = 0.0
        for idx in np.ndindex(x.shape):
            if isinstance(h, dict) and name in h:
                step_h = h[name]
            elif name in TIME_LEAVES:
                step_h = h_time
            else:
                step_h = h_leaf * size[idx]
            xp, xm = x.copy(), x.copy()
            xp[idx] += step_h
            xm[idx] -= step_h
            col = (run(put(inp, path, xp)) - run(put(inp, path, xm))) / (xp[idx] - xm[idx])
            Jc = Jb[(slice(None),) + idx]
            worst = max(worst, float(np.max(np.abs(Jc - col))) * size[idx])          # in units of the state change per input of its own size
            scale = max(scale, float(np.max(np.abs(Jc))) * size[idx])
        err = worst / max(1.0, scale)
        errs[name] = err
        tl = _per_leaf(tol, name)
        assert err < tl, f"Jacobian block {name}: relative error {err:.2e} > {tl:.0e}"
    return errs


def check_dense(backend: Backend, z, u, t, dt, p, tol=1e-9) -> dict:
    """dense at t+dt equals the step result; dense at t+dt/2 equals a half-length step (all blocks)."""
    res = backend.step(z, u, t, dt, p)
    dn = res.dense(np.array([t + dt / 2, t + dt]), partials=True)
    zt, Jt = dn.z, dn.J
    assert res.dense(np.array([t + dt / 2, t + dt])).J is None, "partials=False must not return a Jacobian tree"
    half = backend.step(z, u, t, dt / 2, p)
    e_end = float(np.max(np.abs(np.asarray(zt[1]) - np.asarray(res.z))))
    e_half = float(np.max(np.abs(np.asarray(zt[0]) - np.asarray(half.z))))
    e_J_half = e_J_end = 0.0
    for path, leaf in leaves(Jt):
        e_J_half = max(e_J_half, float(np.max(np.abs(np.asarray(leaf)[0] - np.asarray(get(half.J, path))))))
        e_J_end = max(e_J_end, float(np.max(np.abs(np.asarray(leaf)[1] - np.asarray(get(res.J, path))))))
    assert max(e_end, e_half, e_J_half, e_J_end) < tol, (e_end, e_half, e_J_half, e_J_end)
    return {"end_state": e_end, "half_state": e_half, "half_J": e_J_half, "end_J": e_J_end}


def check_chaining(backend: Backend, z, u, t, dt1, dt2, p, tol=1e-5) -> dict:
    """step(t, dt1+dt2) equals step(t, dt1) followed by step(t+dt1, dt2) with zero controls, and the blocks compose by
    section 12: J["z"] = Jb_z Ja_z, J["p"] = Jb_z Ja_p + Jb_p, J["t"] = Jb_z Ja_t + Jb_t, J["dt"] = Jb_dt.

    Only impulses (role "delta_v", applied at the start of the step) may be non-zero in `u`: a control spread over the step, such as
    a piecewise acceleration on sub-intervals of dt, is not the same control on the longer step, so the identity does not hold.
    Errors are relative to max(1, max |reference|) of each quantity."""
    d = backend.describe()
    for c in d.controls:
        if c.role != "delta_v" and c.name in u and np.any(np.asarray(u[c.name]) != 0):
            raise ValueError(f"check_chaining: control {c.name!r} (role {c.role!r}) must be zero; only impulses are chained")
    whole = backend.step(z, u, t, dt1 + dt2, p)
    a = backend.step(z, u, t, dt1, p)
    b = backend.step(a.z, zero_controls(d), t + dt1, dt2, p)
    A = {k: np.asarray(v) for k, v in a.J.items() if k not in ("u", "p")}
    B = {k: np.asarray(v) for k, v in b.J.items() if k not in ("u", "p")}

    def rel(ref, got):
        ref, got = np.asarray(ref, float), np.asarray(got, float)
        return float(np.max(np.abs(ref - got))) / max(1.0, float(np.max(np.abs(ref))))

    errs = {"state": rel(whole.z, b.z),
            "stm": rel(whole.J["z"], B["z"] @ A["z"]),
            "t": rel(whole.J["t"], B["z"] @ A["t"] + B["t"]),
            "dt": rel(whole.J["dt"], B["dt"])}
    for name, Jw in whole.J.get("p", {}).items():
        errs["p." + name] = rel(Jw, np.tensordot(B["z"], np.asarray(a.J["p"][name]), axes=1) + np.asarray(b.J["p"][name]))
    bad = {k: v for k, v in errs.items() if v >= tol}
    assert not bad, f"chaining: errors above {tol:.0e}: " + ", ".join(f"{k} {v:.2e}" for k, v in bad.items())
    return errs
