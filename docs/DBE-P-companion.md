# DBE-P companion: one back-end and one consumer

Companion to `DBE-P.md` (draft v0.8). It describes how a specific numerical propagator (didi) is made a DBE-P back-end, and how a specific consumer (gogo) uses back-ends. It is not part of the protocol. Section numbers of DBE-P.md are referred to as "DBE-P section N".

## 1. The didi adapter

`didi.dbep.Adapter(spec, state=StateRequest(...), differentiate=(...))` makes a didi `Spec` a back-end.

- **Native state from the spec.** `state=("r","v")` gives the spacecraft slices relative to `spec.central`. `bodies={"EROS": IntegratedBody(central="SUN", perturbers=..., gm="state"|"param")}` adds `r:EROS`, `v:EROS` (centre `SUN`) and optionally `gm:EROS`.
- **Requested coordinates.** The adapter converts to the `StateRequest` using didi's ephemeris provider, for example spacecraft position relative to `EARTH` while integrating relative to `EROS`. It includes the time terms in `J["t"]` and `J["dt"]` (DBE-P section 8.2). A centre it has no ephemeris for is refused at construction. `meta.ephemeris` names the ephemeris (for example `"de440"`).
- **Conversion toolkit.** `dbep.Converted(native, request, ephemeris)` (DBE-P section 8.2, rule 9) does the conversion. The didi adapter can therefore be written natively and wrapped, or can call the same conversion internally. It supplies an `EphemerisProvider` built on its Chebyshev ephemeris, which is differentiable in `t`. A back-end such as gogo's Kepler arc uses the same wrapper.
- **Orientation.** The adapter declares `capabilities.orientation` and implements `orientation(body, t, p)` with `dbep.IAUOrientation` (periodic terms later) over the `orient:<body>:<sub>` parameters, so the dynamics and the orientation service read the same `p`.
- **Controls from the spec:** `Impulse("dv", frame="RTN")`, `PiecewiseAccel("acc", n=6, frame="RTN")`. `PiecewiseThrust` and `m` later.
- **What is differentiated:** chosen when the adapter is built, e.g. `differentiate=("cr_a_m", "gm:EROS")`. The default is every declared parameter. All controls are always differentiated, so a model should not declare controls it does not need.
- **`step` and `dense`:** forwarded to `Propagator.step`. didi owns the variational equations **[DECIDED]**. Adaptive stepping with a padded step buffer provides `dense`; with `partials=False` it skips the variational interpolation.
- **Integrated bodies:** the asteroid's state is propagated together with the spacecraft. The native `J["z"]` is block lower triangular (no back-reaction on the asteroid in v1; the asteroid's motion does not depend on its own GM). The adapter MAY exploit that structure.
- **Provider mode** (SPK or Kepler wrapper) is for bodies treated as known, such as planets and the Sun. Bodies the consumer estimates are integrated.
- **Static states:** a named scalar such as `gm:EROS` or `cr_a_m` can be declared in `state` or passed in `p`, with identical numbers (DBE-P section 12, rule 5).

Build order for the adapter:
1. `step` with `("r","v")`, `Impulse`, `PiecewiseAccel`, central gravity 8×8, third bodies, `cr_a_m` SRP; `describe()`, `meta`, and the `StateRequest` for the native centre.
2. `dense` with Hermite interpolation of `Φ` and sensitivity blocks, and the states-only path.
3. Conversion to other centres, with time terms; refusal of unsupported requests.
4. Integrated asteroid and statics in the state.
5. Spacecraft mass `"m"`, `PiecewiseThrust`, and finite burns.
6. Later: tumbling rotation, numeric small-body centre dynamics beyond v1, reverse mode.


## 2. gogo as consumer

- At model construction gogo builds a `StateRequest` and a `Requires`, builds the back-end with the request, calls `check`, and binds its parameter registry (names, priors, scaling, which are estimated) to the descriptor's names. Scaling of badly conditioned slices (GM against position in km) is gogo's registry's job.
- gogo chooses the coordinates it wants (for example an Earth-centred spacecraft position and an asteroid state relative to the Sun) and expresses its priors and process noise in them. It needs no ephemeris to convert states.
- gogo forms measurement geometry from `z` (station-relative quantities and the like) and, for landmarks and lidar, from `backend.orientation(body, t, p)` (quaternion and its parameter derivatives) and composes chains, sensitivities and observation Jacobians by the rules in DBE-P section 12.
- gogo's Kepler arc becomes a back-end implementing the same protocol and honouring the same `StateRequest`, using the shared conversion toolkit. That makes it comparable against the numeric back-end (conformance) and swappable.
- gogo never integrates a numeric segment and never differentiates through a back-end's integrator. Its own gradients (pass duration, observation accuracy, missing passes) pass through its information accumulator, with the returned Jacobians as constants.
- Latency, passes, weights and station geometry stay in gogo.
