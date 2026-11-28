# DBE-P: dynamics back-end protocol

Status: draft v0.7 (proposal, for review)
Reference code: `dbep.py` (types, requirement check, conformance suite, conversion wrapper `Converted`, quaternion and orientation toolkit), `example_oscillator.py` (toy back-end and consumer), `test_dbep.py` (tests of the toolkit)

Toolkit change in `dbep` 0.7.2 (protocol unchanged): `check_dense` measures errors relative to each block, and at the half step compares impulses only (other controls must be zero), as `check_chaining` does. Before, one absolute tolerance covered blocks of very different sizes, and a correct back-end with a piecewise acceleration failed.

Toolkit change in `dbep` 0.7.1 (protocol unchanged): `Converted` takes a shorter path when no centre of a conversion is an entity of `z`. The conversion is then a translation by an ephemeris state, so its state Jacobian is the identity, and the native state is `z − b(t)`, not the result of a solve. The results are the same, and the cost at every step and dense epoch is lower.

Changes from DBE-P v0.6 (minor: an optional field):
- **Control frames (sections 4, 9, 11, 14).** A control declares the frame of its components: `Control.frame` is `"ICRF"` (the default), `"RTN:<CENTRE>"` or `"VNC:<CENTRE>"`, the local frame of the state relative to CENTRE before the control acts. Named with its centre, it stays defined after `Converted` changes the centre of the state. `Requires.control_frames` states the frame a consumer needs and `check` refuses a mismatch. Before, an impulse in RTN and one in ICRF had identical descriptors, so a consumer could be given the wrong one silently.

Correction in `dbep` 0.6.2 (the protocol stays `DBE-P/0.6`):
- **Scale-free finite-difference errors (section 13).** 0.6.1 divided the error of a block by `max(1, max |J|)`, so a block of small derivatives with respect to a large parameter (the GM of the Sun in an orbit model) was checked absolutely and could be wrong by a factor of two and pass. Columns are now scaled by the size of their input.

Corrections in the reference toolkit `dbep` 0.6.1 (the protocol stays `DBE-P/0.6`; the text changes below only clarify it):
- **Finite-difference steps and tolerances (section 13).** `check_step_jacobians` perturbs the epoch inputs `t` and `dt` by an absolute step (`h_time`, default 1e-3 s). The earlier step, relative to `|t|`, was about 850 s at a 2026 epoch and failed a correct `J["t"]`. Other inputs are stepped relative to the size of their state slice or input array, not of the single component. Errors are relative to the size of each block, and `h` and `tol` can be given per block.
- **Chaining (section 13, test 6)** also checks the composition of `J["p"]`, `J["t"]` and `J["dt"]` (section 12), and refuses non-impulsive controls, for which the identity does not hold.
- **Version check (section 16).** `check` compared only the protocol name, so any version passed. It now refuses a different major version and, while the major version is 0, a different minor version.
- **Results cross `jax.jit` (section 3).** `StepResult` and `DenseResult` are JAX pytrees, with `meta` as static data. The dense closure is not carried across a jit boundary: call `dense` inside the jitted function.
- `check_meta` checks the row count of every block. `Converted` refuses a centre requested for a slice that is not a position or velocity. `Descriptor` is hashable.
- Clarified: the status bitmask (section 10) and the coordinates of process noise (section 12, rule 4).

Changes from DBE-P v0.5:
- **Conversion wrapper.** `dbep.Converted(native_backend, state_request, ephemeris)` implements section 8.2 once, for any back-end that has a native `step`. A back-end needs no conversion code of its own. It needs an `EphemerisProvider` (section 8.2, rule 9). This settles where the conversion toolkit lives (it is in `dbep`).
- **Orientation capability (section 8.6).** A back-end MAY provide `orientation(body, t, p)`, returning a unit quaternion (scalar-first, body-fixed to ICRF) with its derivatives with respect to `t` and the differentiated orientation parameters. `dbep` provides quaternion helpers and a reference IAU service, `IAUOrientation`, that a back-end and a consumer can share. New descriptor capability `orientation`, new `Requires.orientation`, new conformance test 10.
- `Requires` also checks control names, parameter names and that estimated parameters are differentiated.
- `meta.jacobian` lists each leaf with its input shape `s`; the block has shape `(n_z,) + s` (section 6).
- Open questions renumbered (orientation is closed; the toolkit question is narrowed).

Changes from DBE-P v0.4 (carried from v0.5):
- Units are decided: km, s, kg and units derived from them (section 14). Anything defined in metres or Newtons must be converted at the boundary.
- Name syntax: `:` separates one or more qualifiers (`gm:EROS`, `orient:MOON:ra0`). Commas are not used in names. Each orientation parameter is its own scalar parameter.
- Epoch derivatives: `J["t"]` and `J["dt"]` are partial derivatives with respect to the step's own inputs. Dependence of a step's start epoch on earlier steps (a boundary epoch, a manoeuvre time) is composed by the consumer, new section 12 rule 7.
- Open questions renumbered (the units question is closed).

Changes from DBE-P v0.3 (carried from v0.4):
- The consumer specifies the coordinates it expects in `z` (centre, axes, unit per slice) in the model specification, as a `StateRequest`. The back-end MUST return exactly that, converting internally if its native coordinates differ, so that every Jacobian is the derivative of the requested-coordinates state, or refuse at construction. This reverses v0.3 section 8.2 ("the consumer adapts").
- Result metadata (`meta`) stays, so results are self-describing; it now reports the realised coordinates, which equal the request.
- `Requires` carries the `StateRequest`; `check` compares the realised state with it.
- Ephemeris consistency is mostly an internal matter of the back-end now.

Earlier: v0.3 removed `outputs` and level 3 and added `meta`; v0.2 removed `wrt` (every Jacobian block is returned, what is differentiated is fixed by the model) and added the `partials` flag to `dense`.

Keywords MUST, SHOULD, MAY are used in the usual sense. Items marked **[OPEN]** are undecided.

---

## 1. Purpose

DBE-P defines how a consumer asks a dynamics model for state transitions and their derivatives, without knowing how the model is built. A consumer needs, for each trajectory segment:

1. the end state,
2. first-order sensitivities of the end state to everything it may estimate or analyse (initial state, controls, parameters, time),
3. the same quantities at arbitrary epochs inside the segment (observation times).

The consumer specifies what its state vector must be (for example spacecraft position and velocity relative to Earth, in ICRF axes). The back-end delivers the state in exactly those coordinates, converting from its native ones if they differ. The consumer therefore needs no knowledge of the back-end's internal centres, layout or ephemeris to chain steps and form measurement geometry.

The protocol has a domain-neutral **core** (sections 3 to 13) and **profiles** (section 14 on) that give names and conventions a physical meaning. A consumer that needs orbital state requires the core plus the `orbital/1` profile.

## 2. Roles and terminology

| Term | Meaning |
|---|---|
| **back-end** | any implementation of DBE-P (a numerical propagator, an analytic Kepler model, a linear test model) |
| **adapter** | the code that makes an existing library a back-end. For code in another language, a **binding**. |
| **consumer** | code that uses a back-end (an orbit-determination covariance tool, an estimation library, a guidance tool) |
| **model** | one configured back-end instance (for example an Eros-proximity model) |
| **descriptor** | the static description a back-end publishes about its model |
| **meta** | the part of the descriptor that describes the state and the Jacobian tree, attached to every result |
| **profile** | a set of conventions on top of the core (axes, units, role names, id space of centres) |

## 3. Core conventions

- **Pure functions.** `step` MUST be pure: the same inputs give the same results, with no hidden state between calls.
- **Array types.** Inside jit, all leaves are JAX arrays and all dict keys are static strings; `meta` is static. Conversion to numpy happens only at the outermost boundary of the consumer. Back-ends not written in JAX MAY accept and return numpy arrays, and then do not claim `jit` in their capabilities.
- **Precision.** float64.
- **Names.** Names of state slices, controls and parameters are strings. They MUST NOT contain `.` or `,`. The character `:` separates qualifiers: first a body or an object, then optionally a sub-quantity, e.g. `gm:EROS`, `orient:MOON:ra0`.
- **Input tree.** `{"z": array, "u": {name: array}, "t": scalar, "dt": scalar, "p": {name: array}}`.
- **Paths.** Jacobian leaves are addressed by the same dotted paths as the inputs: `"z"`, `"u.dv"`, `"p.k"`, `"t"`, `"dt"`.
- **Batch axis.** Where stated, a leading batch or epoch axis MAY be added to array leaves. A back-end declares support in `capabilities.batch`.
- **Results under jit.** `StepResult` and `DenseResult` are pytrees: their arrays are the leaves and `meta` is static, so a result can be returned from a jitted function. The dense output of a step is a closure, not an array, and is not carried out of a jitted function: call `res.dense(...)` inside it.

## 4. Descriptor

A back-end publishes `describe() -> Descriptor` with:

| Field | Content |
|---|---|
| `protocol` | `"DBE-P/0.7"` |
| `profile` | e.g. `"orbital/1"` |
| `time` | scale, epoch and unit of `t` and `dt` |
| `state` | ordered slices: `name`, `size`, `role`, `unit`, `axes`, `centre` |
| `controls` | `name`, `shape`, `role`, `unit`, `frame` |
| `params` | `name`, `shape`, `default`, `unit`, `differentiated` |
| `ephemeris` | id of the ephemeris and constants set the model uses, `""` if none |
| `capabilities` | `level` (0 to 2), `dense`, `batch`, `jit`, `reverse`, `orientation` |
| `status_bits` | meaning of each status bit (section 10) |

The descriptor reports the *realised* state: when a `StateRequest` was given at construction, the slices' `centre`, `axes` and `unit` equal what was requested (section 8). The descriptor is static. It MUST NOT depend on the values passed to `step`. `n_z` is the sum of the slice sizes, and slices are laid out in z in the order listed. `descriptor.state_meta()` yields the `meta` object (section 8).

## 5. `step`

```python
res = backend.step(z, u, t, dt, p)
```

- `z`: state, shape `(n_z,)`, layout and meaning from `meta`.
- `u`: dict of controls named in the descriptor. A missing control means zero.
- `t`: start epoch. `dt`: duration (MAY be traced, MAY be negative if the back-end declares it).
- `p`: dict of parameters. A missing parameter takes the descriptor's default.
- There is no per-call selection of derivatives. The Jacobian tree is fixed by the model (section 6).

Result:

```python
res.z        # (n_z,) state at t + dt
res.J        # Jacobian tree, section 6
res.status   # int bitmask, section 10
res.meta     # StateMeta, section 8
res.dense    # callable, section 7 (level >= 1)
```

Internal sub-stepping is the back-end's business. A step may span days and contain thousands of integrator steps. Controls take effect inside the step as defined by the control's role in the profile.

## 6. Jacobian tree

`res.J` mirrors the inputs. A back-end returns **every** block, with no selection at call time:
- `"z"`,
- every control in `"u"`,
- every parameter in `"p"` whose descriptor entry has `differentiated=True`,
- `"t"` and `"dt"`.

What is differentiated is decided when the model is built (for example an adapter option) and is visible in the descriptor. A parameter with `differentiated=False` is held fixed at the value passed in `p` and has no leaf. The cost of a call scales with the number of Jacobian columns, so a model SHOULD flag only the parameters a consumer actually estimates. Because the tree is fixed by the model, a consumer knows its structure before the first call, and a jitted function compiles once per model. `meta.jacobian` lists every leaf with its input shape `s` (not the block shape): the block of a leaf is `(n_z,) + s`, and `n_z` is in `meta`.

For every leaf of shape `s`, the block has shape `(n_z,) + s`:

```
res.J["z"]               (n_z, n_z)      state transition matrix
res.J["u"]["dv"]         (n_z, 3)
res.J["p"]["k"]          (n_z,)
res.J["t"]               (n_z,)
res.J["dt"]              (n_z,)
```

Definitions:
- `J["z"]`: derivative of the end state with respect to the start state.
- `J["u"][c]`, `J["p"][q]`: derivative with respect to a control or parameter.
- `J["dt"]`: derivative with respect to the duration at fixed start. This is the flow derivative at the end state.
- `J["t"]`: derivative with respect to shifting the whole step in time at fixed `dt`. It captures explicit time dependence (ephemerides, orientation, time-tagged forces).

`J["t"]` and `J["dt"]` are partial derivatives with respect to this step's own inputs `t` and `dt`, treated as independent. They do not include how `t` depends on earlier steps. When start epochs are linked (the end of one step is the start of the next), the consumer composes the derivative with respect to a boundary epoch (section 12, rule 7).

All Jacobians are relative to the **start** of the step, never to a previous epoch. Composition across steps is the consumer's job (section 12).

## 7. Dense output

```python
d = res.dense(t_vec)                    # states only
d = res.dense(t_vec, partials=True)     # states and Jacobians;  t_vec: epochs in [t, t+dt]
```

- `d.z`: `(n_t, n_z)`. `d.meta`: the same `meta` as the step result.
- `d.J`: `None` without `partials`. With `partials=True`, the same tree structure as `res.J`, each leaf with a leading `n_t` axis.
- Jacobians are relative to the start of the step.
- `d.J["dt"]` is the sensitivity to the evaluation offset `tau = t_i - t`, i.e. to the observation time tag. `d.J["t"]` is `J["t"]` of a step of duration `tau`: shifting the start at fixed `tau`.
- `partials` is a static Python bool; changing it may recompile. Without it a back-end SHOULD skip all variational work for the interpolation, which is the cheap path for consumers that need only states.
- `dense` MUST be pure, jittable and vmappable if the back-end claims `jit`.
- Accuracy: interpolated values MUST be consistent with the end values (dense at `t+dt` equals `res.z` and `res.J`). Interpolation order and error are the back-end's responsibility and SHOULD be stated in its documentation. **[OPEN]** interpolation scheme and whether variational states enter step-size control.

## 8. State specification, conversion and metadata

### 8.1 The consumer specifies the coordinates

The model specification includes a `StateRequest`: for each slice, optionally a `centre`, `axes` and `unit`. An empty field means "the back-end's choice", which it then declares.

```python
want = StateRequest(slices=(SliceRequest("r", centre="EARTH", axes="ICRF", unit="km"),
                            SliceRequest("v", centre="EARTH", axes="ICRF", unit="km/s")))
backend = make_backend(model_spec, state=want)   # the constructor is back-end specific and MUST accept a StateRequest
check(Requires(profile="orbital/1", level=2, state=want), backend.describe())
```

The back-end MUST return `z` in exactly those coordinates: the input state, `res.z`, the states of `dense`, and all Jacobians. If its native coordinates differ, it converts internally. If it cannot provide a request, it MUST refuse when the model is built, with `DBEPUnsupported` naming what it cannot do. `check` compares the realised descriptor with the request, which catches a back-end that ignores it.

### 8.2 Conversion rules

**The converted step.** Let `F(z, t, dt)` be the back-end's native step, written in its own coordinates, and let `T(t)` convert a native state to the requested coordinates at epoch `t`. The step in the requested coordinates is

```
F_req(z_req, t, dt)  =  T(t + dt)( F( T(t)⁻¹(z_req), t, dt ) )
```

Read it from the inside out: convert the incoming requested-coordinates state to native coordinates at `t`; take the native step; convert the result to requested coordinates at `t + dt`. `T` is evaluated at two different epochs, because a centre may move during the step.

For a change of centre, `T` is a translation: `T(t)(z) = z + m(t)`, where `m(t)` is the state (position in the position rows, velocity in the velocity rows) of the native centre relative to the requested centre.

**Rules**

1. **The Jacobians are those of the requested state.** Every block the back-end returns is the derivative of the requested-coordinates end state (the output of `F_req`) with respect to the requested-coordinates start state, the controls, the parameters, `t` and `dt`. The consumer never sees native coordinates.
2. **Time dependence of `T`.** If `T` depends on time (a moving centre), `J["t"]` and `J["dt"]` contain terms the native step does not have. For a translation, with `m'` the time derivative of `m` (velocity in the position rows, acceleration in the velocity rows):

   ```
   J["z"]  = J_native["z"]
   J["dt"] = J_native["dt"] + m'(t + dt)
   J["t"]  = J_native["t"] − J_native["z"] · m'(t) + m'(t + dt)
   ```

   Derivation: write `F_req(z_req, t, dt) = F(z_req − m(t), t, dt) + m(t + dt)` and differentiate. For `J["dt"]` hold `z_req` and `t` fixed: only the `dt` argument of `F` and `m(t + dt)` depend on `dt`. For `J["t"]` hold `z_req` and `dt` fixed: the native start state `z_req − m(t)` moves by `−m'(t)` (pushed through `J_native["z"]`), `F` has its own explicit `t` dependence (`J_native["t"]`), and `m(t + dt)` contributes `m'(t + dt)`. Control and parameter blocks equal the native ones, provided the control is defined in axes that `T` does not change (an inertial `dv`, for example).

   For a `T` that also rotates the axes, `J["z"]` gets the state Jacobians of `T` at both epochs (`∂T(t+dt)/∂z · J_native["z"] · ∂T(t)⁻¹/∂z`), and the time terms involve `∂T/∂t` as well. A back-end MAY compute these analytically or by automatic differentiation of `F_req`, as long as they are the derivatives of `F_req`.
3. **Example.** A native centre moves at constant velocity `vc` relative to the requested centre, and the model has no explicit time dependence. Then `m' = [vc; 0]` at all times, and for a free particle `J_native["z"] · m' = [vc; 0]`. The result is `J["t"] = 0 − [vc; 0] + [vc; 0] = 0`: shifting the whole step in time changes nothing, as it should. `J["dt"]` is the native one plus `vc` in its position rows: a longer step moves the centre further, so the converted end position changes by that much more than the native one.
4. **The conversion is an invertible change of coordinates** of the native state at a given epoch. Each entity keeps one position slice and one velocity slice. A centre may be any body whose state the back-end knows (from its ephemeris), or another entity in `z`. Cycles (A relative to B and B relative to A) are not allowed.
5. **Conversion happens at the boundaries of a step and at dense epochs.** Integration stays in the back-end's native coordinates. Conditioning and numerical choices are the back-end's business.
6. **Axes.** `orbital/1` allows `ICRF` only. Non-inertial axes need the profile to define "velocity" first (inertial velocity expressed in those axes, or rate of change within them). **[OPEN]**
7. **Ephemeris.** The back-end uses its own ephemeris for conversions and names it in `meta.ephemeris`. The consumer needs no copy of it for state conversion. A consumer that uses its own ephemeris for anything that must agree with the back-end's SHOULD compare the ids.
8. **Typical conversions** for a position `r` relative to a native centre `A`, requested relative to a centre `C`:

   | Native state | Conversion the back-end performs | state Jacobian of the conversion |
   |---|---|---|
   | centre `A` is `C` | none | `I` |
   | `A` is a body with a known ephemeris | `r + r_A(t) − r_C(t)` | `I`; time terms `v_A − v_C` in the position rows of `J["t"]`, `J["dt"]` |
   | `A` is an entity whose state is in `z` | `r + r_A − r_C(t)`, with `r_A` taken from `z` | `[I  I  0]`; time term `−v_C` |

9. **Reference implementation.** `dbep.Converted(native, request, ephemeris)` implements section 8.2 for translations of centre, so that a back-end only has to provide its native `step`. It computes `z_req`, every Jacobian block, the time terms (`m'` is obtained by automatic differentiation of the conversion, so the ephemeris only has to be differentiable in `t`), the dense states and the dense Jacobians. The ephemeris interface is
   ```python
   class EphemerisProvider(Protocol):
       id: str                                   # goes into meta.ephemeris
       def state(self, body: str, t) -> Array    # (6,) position [km], velocity [km/s] relative to the SSB, ICRF, TDB
   ```
   A centre is `"SSB"`, a body the provider knows (it raises `KeyError` otherwise), or an entity in `z`. The wrapper refuses (`DBEPUnsupported`) a body it cannot resolve, a cycle, non-ICRF axes, a unit it does not convert, a request that gives `r` and `v` different centres, and an entity used as a centre that is itself re-centred in the same request.

### 8.3 Metadata

Every result (`StepResult`, `DenseResult`) carries `meta`, a static object:

| Field | Content |
|---|---|
| `slices` | for each slice of z, in order: `name`, `size`, `role`, `unit`, `axes`, `centre` |
| `time` | scale, epoch and unit |
| `ephemeris` | id of the ephemeris/constants set the model uses |
| `jacobian` | every leaf of the Jacobian tree and its shape |
| `protocol`, `profile` | versions |

`meta.index(name)` gives the slice of `z` for a name, and `meta.slices_with_role(role)` finds slices by role. A back-end MUST return the same `meta` that `describe()` gives. Results stay self-describing when stored or passed around without the back-end, and a consumer can verify that what it received is what it asked for.

### 8.4 What the consumer does

1. **Chooses its coordinates** to suit its measurements (for example an Earth-centred position for range and Doppler) and states them in the model specification.
2. **Expresses priors and process noise** in those coordinates. Nothing needs mapping, because it chose them.
3. **Chains across back-ends:** every back-end honouring the same `StateRequest` shares the same coordinates. Slices are matched by name and role. A slice present on one side only is carried as constant (identity Jacobian) or dropped if it is not estimated.
4. **Forms measurement geometry beyond the state** (station-relative quantities, for example) from `z` and its own models. Where `z` is already what the measurement needs, the geometry is a selection of `z`.

### 8.5 What this does not cover

Orientation of a body that is part of `z` (tumbling, rotation integrated with the dynamics) is not covered. Orientation that is a function of time and parameters is covered by the capability in section 8.6.

### 8.6 Orientation capability

Observations that depend on a body's orientation (landmarks, lidar, body-fixed stations, forces in body-fixed axes) need the rotation from body-fixed to ICRF axes and its derivatives with respect to the estimated orientation parameters. A back-end that has an orientation model declares `capabilities.orientation = True` and provides

```python
o = backend.orientation(body, t, p)     # t: TDB s past J2000; p: parameters as for step
o.q                                     # (4,) unit quaternion, scalar-first (w, x, y, z), Hamilton product
o.J["t"]                                # (4,)       dq/dt
o.J["p"]["orient:MOON:ra0"]             # (4,)       dq/dparameter, for every differentiated orientation parameter
```

- **Convention.** `q` rotates body-fixed components into ICRF components: `v_icrf = q v_body q*`. Quaternions are used, not matrices or angles, because the product and the interpolation are free of singularities and the parameters stay separate. The sign of `q` is not fixed, but `q` and its derivatives are continuous in `t` and in the parameters. A consumer compares orientations through the rotation, not component by component.
- **Dependence.** The orientation is a function of `t` and `p` only, not of `z`. A consumer combines it with the state sensitivities directly: the derivative of an observation with respect to an orientation parameter is the sum of the geometry route (`o.J["p"]`) and, where the parameter also enters the dynamics, the route through `res.J["p"]` (section 12, rule 3).
- **Parameters.** `orient:<body>:ra0`, `:dec0`, `:W0`, `:rate` (section 14). The orientation service and the dynamics of the same back-end read the same values of `p`.
- **Shared reference.** `dbep.IAUOrientation` is a pure JAX implementation of the IAU model with linear terms: `R = Rz(ra0 + 90 deg) Rx(90 deg − dec0) Rz(W0 + rate t)`. A back-end can use it, and a consumer can call it directly, so both agree on the model. Periodic terms are added to the same class later.
- **Helpers.** `quat_mul`, `quat_conj`, `quat_rotate`, `quat_to_matrix`, `iau_quaternion` in `dbep`.

## 9. Controls and parameters

- **Controls** are known inputs of a step. Their schema (name, shape, role, unit, frame) is in the descriptor. `frame` names the axes of the control's vector components (section 14); it is part of the meaning of the control, so a consumer that applies or perturbs a control states the frame it expects (`Requires.control_frames`).
- **Parameters** are named values. A parameter flagged `differentiated` appears in `J["p"]`. One flagged otherwise is held fixed at the value in `p`. One absent from `p` takes the default.
- A name MUST NOT appear both as a state slice and as a parameter. A back-end MAY offer the same named scalar in either place (state or parameter) depending on its configuration; the numbers are identical (section 12, rule 5).
- Process noise (random accelerations, manoeuvre execution error) is the consumer's business. It uses `J["u"]` to map control uncertainty and the slice roles (section 11) to inject noise.

## 10. Status

`res.status` is an int bitmask per batch member. `status == 0` means no condition was raised. Bit `k` is the value `1 << k`, so `max_steps` alone gives `status == 2`. Bit 0 is reserved and never set. `status_bits` in the descriptor maps the bit number `k` to its name.

| Bit | Meaning |
|---|---|
| 0 | reserved |
| 1 | `max_steps` exceeded |
| 2 | step-size underflow |
| 3 | outside coverage (ephemeris, orientation) |
| 4 | impact |
| 5 | non-finite result |

A back-end MUST NOT raise inside jitted code. The consumer checks `status` after the call and decides to raise, mask or warn. A back-end MAY extend the table in `status_bits`.

## 11. Capability levels and requirements

| Level | Provides |
|---|---|
| 0 | `step` with `J["z"]` |
| 1 | + `dense` |
| 2 | + the `u`, `p`, `t`, `dt` blocks |

A consumer states what it needs:

```python
Requires(profile="orbital/1", level=2, roles=("position", "velocity"), state=want, dense=True,
         controls=("dv",), params=("cr_a_m",), estimated=("cr_a_m",), orientation=False, control_frames=(("dv", "ICRF"),))
check(requires, backend.describe())      # raises DBEPMismatch naming every unmet item
```

`controls` and `params` must be declared by the model, `estimated` parameters must also be differentiated, `orientation=True` requires the orientation capability (section 8.6), and every `(control, frame)` in `control_frames` must be a declared control in that frame.

`check` MUST be called when a model is constructed, so mismatches appear before any propagation. For every slice in the `StateRequest` it verifies that the realised `centre`, `axes` and `unit` equal what was requested.

**Roles.** Each state slice, control and parameter has a role (`position`, `velocity`, `mass`, `gm`, `delta_v`, ...). Roles let a consumer act on a slice without knowing its name or the back-end. For example, a consumer injects process noise into the slice with role `velocity` without knowing the slice's name.

## 12. Composition rules (what a consumer does with the results)

Let step `k` start at `t_k` with state `z_k`, and let `Φ_k = res_k.J["z"]`.

1. **Chaining steps:** `Φ_total = Φ_k · … · Φ_1`. A parameter sensitivity accumulates as `S_k = Φ_k S_{k-1} + res_k.J["p"][name]`. A control sensitivity maps through `Φ` in the same way.
2. **Inside a step:** the state sensitivity at epoch `t_i` with respect to the initial state of the chain is `d.J["z"](t_i) · Φ_{total, previous steps}`.
3. **Observations.** With `obs = g(G(z, t))`, where `G` is the consumer's own mapping from the state to geometry (a selection of `z` when the state is already in the coordinates the measurement needs) and `g` the measurement model:

   ```
   ∂obs/∂z0 = ∂g/∂G · ∂G/∂z · d.J["z"](t_i) · Φ_{previous}
   ∂obs/∂p  = ∂g/∂G · ∂G/∂z · [ d.J["z"](t_i) · S_{previous} + d.J["p"](t_i) ]
   ```

   For a time-tag error: `∂obs/∂t_i = ∂g/∂G · ( ∂G/∂t|explicit + ∂G/∂z · d.J["dt"](t_i) )`. For Doppler the consumer uses both a position and a velocity slice.
4. **Process noise** is added by the consumer at step boundaries, in the requested coordinates (section 8.1), which are the coordinates of `z`.
5. **State/parameter equivalence:** a named scalar declared as a state slice gives `J["p"][name]` of a parameter-based model as the corresponding column of `J["z"]`, restricted to the dynamic rows. The row of that slice in `J["z"]` is the unit vector.
6. **Mixed back-ends in a chain** (for example an analytic Kepler arc followed by a numeric arc): both honour the same `StateRequest` (section 8.4, point 3), so layout and coordinates agree. A slice present on one side only is carried or dropped by the consumer.

7. **Epoch sensitivities** (boundary epochs and manoeuvre times). Steps `k` and `k+1` are linked: the end of step `k` is the start of step `k+1`, with epoch `T_k`, so `dt_k = T_k − T_{k−1}` and `dt_{k+1} = T_{k+1} − T_k`. An impulse at the start of step `k+1` moves with `T_k`. Moving `T_k` with `T_{k−1}` and `T_{k+1}` fixed changes `dt_k` by `+1`, `t_{k+1}` by `+1` and `dt_{k+1}` by `−1`, so
   ```
   ∂z_{k+1}/∂T_k = J_k["dt"]
   ∂z_{k+2}/∂T_k = Φ_{k+1} J_k["dt"] + J_{k+1}["t"] − J_{k+1}["dt"]
   ```
   and later states follow by multiplying by the further `Φ`. For an observation at a fixed epoch `t_i` inside step `k+1` (evaluation offset `tau_i = t_i − T_k`, which shrinks as `T_k` grows):
   ```
   ∂z(t_i)/∂T_k = d.J["z"](t_i) · J_k["dt"] + d.J["t"](t_i) − d.J["dt"](t_i)
   ```
   The toy consumer in `example_oscillator.py` checks this against finite differences.

## 13. Conformance (shared by all back-ends; implemented in `dbep.py`)

Back-end tests:
1. **Meta consistency:** `res.meta` equals the descriptor's meta; `z` has length `n_z`; the Jacobian tree has exactly the leaves in `meta.jacobian`.
2. **Requested coordinates:** `check` passes with the `StateRequest` the model was built with; an unsupported request is refused at construction with `DBEPUnsupported`.
3. **Finite differences:** every leaf of the returned Jacobian tree against central differences of `step`, to a stated tolerance, on the model *as built with the request* (so the time terms from a moving centre are exercised).
   - *Steps.* The epoch inputs `t` and `dt` take an absolute step (`h_time`, default 1e-3 s). An epoch is an offset from J2000, so its size says nothing about the time scale of the dynamics. Every other component takes `h · max(1, max |x|)`, the maximum taken over its state slice (for `z`) or its whole input array, so a near-zero component of a large vector does not get a tiny step. The quotient divides by the step actually taken in float64.
   - *Errors* are scale-free: each column is multiplied by the size `s` of its input (the `max(1, max |x|)` of the steps; 1 s for `t` and `dt`), so that `J s` is the state change for a change of the input of its own size, and the error of a block is `max |FD − J| s` divided by `max(1, max |J s|)`. A parameter with a large value and a small derivative (the GM of the Sun) is then checked relative to its effect. A back-end states its tolerance, per block where needed (`tol={"default": ..., "p.cr_a_m": ...}`), and tests at an epoch where it is used (around 8e8 s for the 2020s), not only near `t = 0`.
   - *Epoch resolution.* float64 resolves about 1.2e-7 s at 8.5e8 s. A back-end that forms internal times as `t + i·h` carries rounding noise of that order into its results, and that noise is divided by the step in a finite difference. This is why `h_time` is not smaller.
4. **Conversion consistency:** the same physics built natively and with a converted request differ by exactly the conversion `T`; `J["z"]` follows from the Jacobian of `T`.
5. **Dense consistency:** `dense` at `t+dt` equals the step result; `dense` at `t+dt/2` equals a step of half the duration (including its Jacobians with `partials=True`); `partials=False` returns no Jacobian tree. Errors are relative to each block. At the half step only impulses are compared among the controls, and other controls must be zero: a control spread over the step is defined relative to it. The tolerance follows the back-end's accuracy: for a fixed-step integrator, `J["dt"]` of dense output (offset on the step's grid) and of a half-length step (duration, grid scaling) differ by the truncation error.
6. **Chaining:** one step of `dt1+dt2` equals two steps with zero controls in the second, and the blocks compose by section 12: `J["z"] = Φ_2 Φ_1`, `J["p"] = Φ_2 J_1["p"] + J_2["p"]`, `J["t"] = Φ_2 J_1["t"] + J_2["t"]`, `J["dt"] = J_2["dt"]`. Only impulses may be non-zero: a control spread over the step (a piecewise acceleration on sub-intervals of `dt`) is not the same control on the longer step.
7. **State/parameter equivalence** (where a back-end offers both).
8. **Status:** out-of-coverage and impact return the right bit.
9. **Batch:** a batched call equals a loop (where `batch` is claimed).

Consumer-side test (in `example_oscillator.py`): the observation Jacobians built from `dense` and the chain agree with finite differences of the whole pipeline, with no consumer-side conversion.

The consumer test also covers the boundary-epoch derivative of section 12, rule 7. The toy back-end exercises tests 1 to 6 and the consumer tests, with finite-difference errors around 1e-9, and around 1e-7 for `J["dt"]`, which is the truncation error of the 1e-3 s step on the toy's fast dynamics (test 4 is printed in the demo, not asserted). `test_dbep.py` also runs test 3 at a 2026 epoch. Tests 7 to 9 are not exercised by it.

10. **Orientation** (where `capabilities.orientation`): unit quaternion; proper rotation matrix; `quat_rotate` agrees with the matrix; every leaf of `o.J` against central differences (`dbep.check_orientation`).

Tests of the toolkit itself (`test_dbep.py`): the wrapper reproduces a back-end that converts internally (state, every Jacobian leaf, dense states and Jacobians); the wrapper passes tests 1, 3, 5 and 6 for a moving ephemeris body and for an entity in `z` as the centre (where `J["z"]` changes); the refusals; the orientation service, including a pole-direction check; the extended `Requires` check.

## 14. Orbital profile `orbital/1`

Conventions on top of the core for orbital dynamics. This section defines what `z` contains.

**Contents of state slices**

- `position` slice: the position of the spacecraft (or of the named body) relative to the slice's `centre`, in the slice's `axes`, at the epoch of the state (`t` for the input `z`, `t + dt` for `res.z`).
- `velocity` slice: the TDB-time derivative of the corresponding position slice, in the same centre and axes. It is a coordinate velocity: no light time, no relativistic terms.
- **Centre and axes are chosen by the consumer** through the `StateRequest` (section 8). The back-end converts if its native choice differs. If no request is made for a slice, the back-end declares its native choice.
- **Axes:** `"ICRF"` only in `orbital/1`: inertial, non-rotating, ICRF-aligned.
- **Centre:** a body id (SPICE-style names such as `"SUN"`, `"EARTH"`, `"EROS"`) or `"SSB"`. A back-end supports the centres for which it has an ephemeris, plus the integrated bodies in its state.
- **Time:** TDB seconds past J2000 (`meta.time`).
- **Units [DECIDED]:** base units km, s, kg. Derived: km/s (velocity), km/s² (acceleration), km³/s² (GM), kg·km/s² (thrust; 1 N = 1e-3 kg·km/s²), km²/kg (`cr_a_m`, with the area in km² and the mass in kg). Specific impulse is in s and the standard gravity is `g0 = 9.80665e-3 km/s²`, so the impulse mass law below needs no conversion. Metres and Newtons do not appear in the protocol: a consumer or adapter that has them converts at its boundary, and the descriptor's `unit` fields are the check.

**Everything uncertain is visible.** A quantity that is estimated, or whose uncertainty matters to the consumer, MUST appear in `z` or in `p`. Hence:
- A body whose ephemeris carries estimated parameters MUST be an *integrated body* (its slices are in `z`). A request for it as a centre is satisfied from those slices.
- A body in *provider mode* is treated as known. Its positions come from the ephemeris named in `meta.ephemeris`. Provider-mode ephemeris parameters are not differentiated. A request for it as a centre is satisfied from that ephemeris.

**State slice roles and names**

| Name | Size | Role |
|---|---|---|
| `r` | 3 | `position`, relative to `centre` |
| `v` | 3 | `velocity`, relative to `centre` |
| `m` | 1 | `mass`, kg (later) |
| `gm:<body>` | 1 | `gm`; constant dynamics |
| `r:<body>`, `v:<body>` | 3 | an integrated body's position and velocity relative to its own centre |

**Controls**

| Name pattern | Role | Meaning |
|---|---|---|
| impulse (`dv`) | `delta_v` | impulsive manoeuvre applied at the start of the step, before propagation, in the frame the control declares (`Control.frame`: `ICRF`, `RTN:<CENTRE>` or `VNC:<CENTRE>`) **[DECIDED: all three]**. With mass in the state: `m⁺ = m·exp(−|dv|/(Isp·g0))` (dv in km/s, `g0 = 9.80665e-3 km/s²`). Without mass, the impulse is massless. |
| `acc` | `acceleration` | known acceleration, piecewise constant on `n` equal sub-intervals of `dt` (or on fixed fractional knots) |
| `thrust` | `thrust` | thrust with mass flow; requires `m` in the state (later) |

The impulse frame rotation depends on the state, so the back-end includes `∂R/∂z` in `J["z"]`. Control axes are inertial ICRF unless the control declares otherwise.

**Control frames.** `ICRF`: inertial components. `RTN:<C>`: R along the position, N along the orbit normal (r × v), T = N × R. `VNC:<C>`: V along the velocity, N along the orbit normal, C = V × N. Both are evaluated with the state relative to the centre `C` at the instant the control acts (for an impulse, the start of the step, before it is applied). The centre is part of the name because a conversion (section 8.2) may report the state relative to another centre, where the local frame would differ. A conversion does not change a control's frame: `Converted` passes it through.

**Parameters:** `gm:<body>`, `cr_a_m` (lumped SRP coefficient, used without a mass state) or `cr` and `area` (with mass in the state), `orient:<body>:ra0`, `orient:<body>:dec0`, `orient:<body>:W0`, `orient:<body>:rate` (pole and rotation parameters of an IAU model, each a scalar in rad or rad/s, `W0` at J2000; used by the orientation capability and by forces that use the orientation; more sub-quantities may be added after the second colon).

**Process noise injection** by the consumer: add to the slice with role `velocity` (impulse-like) or to a `position`/`velocity` pair (random acceleration over a step), in the requested coordinates.

## 15. Ensembles, jit, differentiability

- A back-end claiming `batch` accepts a leading batch axis on every leaf. Case A: one shared `t` and `dt` across the batch **[DECIDED]**. **[OPEN]** shared or independent adaptive stepping.
- `step` is jittable. The Jacobian tree is fixed by the model, so there is one compilation per model. `dense(partials=...)` is a static flag.
- Forward mode through `step` is supported. Reverse mode (adjoint) is deferred **[DECIDED]**; adaptive `while_loop` integrators are not reverse-differentiable in JAX. `capabilities.reverse` declares it.

## 16. Versioning

`DBE-P/<major>.<minor>`. Adding optional descriptor fields, roles, controls or capabilities is a minor change. Changing shapes, tree structure, Jacobian definitions, axes or units is a major change. While the protocol is a draft (0.x), any change may break. A profile has its own integer version (`orbital/1`). A consumer MUST refuse a back-end whose protocol major version differs. While the major version is 0, it MUST also refuse a different minor version, since any change may break. `dbep.compatible_protocol` implements this rule and `check` applies it.

## 17. Open questions

| # | Question | Needed for |
|---|---|---|
| 1 | Dense interpolation scheme; whether variational states enter step-size control | accuracy, cost |
| 2 | Ensemble stepping: shared or independent | performance |
| 3 | Memory bound of the dense buffer (`max_steps`, number of differentiated columns) | cost |
| 4 | Whether `meta.ephemeris` needs a checksum and coverage interval, for consumers that use their own ephemeris for other purposes | safety |
| 5 | Negative `dt` (backward propagation) in the core or by capability | smoothing |
| 6 | Naming (`step` vs `flow` or `transition`) | API |
| 7 | Whether controls also need a `differentiated` flag, so unused control columns are not computed | cost |
| 8 | Which package provides the production `EphemerisProvider` (didi's Chebyshev ephemeris, or ephemerix), and the interpolation order of its derivatives | architecture |
| 9 | Non-inertial axes (EME2000, body-fixed): definition of "velocity" in them | later profile version |
| 10 | Whether a `StateRequest` may also change the layout (ordering, grouping of slices), or only centre, axes and unit | consumer convenience |

