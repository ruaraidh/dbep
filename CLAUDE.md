# dbep: guide for Claude Code

dbep holds the dynamics back-end protocol (DBE-P) and a small reference toolkit. Two other projects depend on it: **didi** (provides back-ends)
and **gogo** (consumes them). Keep this package small, stable and free of dynamics.

## Rules

* **The spec is the source of truth.** `docs/DBE-P.md` first, code second. A protocol change means: edit the spec (and its "Changes from" list),
  change `dbep.py`, change or add a conformance test, bump `PROTOCOL`. Never change a Jacobian definition, a shape, an axis or a unit without
  saying so in the spec (DBE-P section 16: that is a major change).
* **Dependencies:** numpy and jax only. No import of didi or gogo, ever. Tests use the toy back-ends in `examples/` and `tests/`.
* **Conventions that must hold everywhere:** float64; units km, s, kg and units derived from them (no metres, no Newtons); TDB seconds past J2000;
  names contain no `.` or `,`; `:` separates qualifiers; every Jacobian block is returned, with no per-call selection; Jacobians are relative to the start of the step;
  `step` is pure; a back-end never raises inside jitted code (it sets `status` bits).
* **Quaternions** are scalar-first, Hamilton, body-fixed to ICRF (`v_icrf = q v q*`).
* **Conformance is the product.** A new capability or rule needs a test that fails for a wrong implementation (a finite-difference check, a conversion check against a
  back-end that converts internally, a chaining check). A test that cannot fail is not a test. Add the case that is easiest to get wrong (for example a centre that is an entity in `z`).
* Keep `Converted` general. If a conversion case needs back-end-specific code, the spec is probably missing a rule: raise it, do not special-case it.
* Do not add features because they might be useful. Open questions live in section 17 of the spec and are decided by the user.

## Where things live

| You want | Look in |
|---|---|
| Types, `check`, `Requires`, conformance checks | `dbep.py` (sections marked by comment rules) |
| Centre conversion, ephemeris interface | `Converted`, `EphemerisProvider` in `dbep.py`; spec section 8.2 |
| Orientation, quaternions | `IAUOrientation`, `check_orientation`, `quat_*` in `dbep.py`; spec section 8.6 |
| A worked back-end and consumer | `examples/example_oscillator.py` (the consumer test includes the boundary-epoch derivative, spec section 12 rule 7) |

## Checks before you finish

```bash
python -m pytest -q                      # all tests, about 1 minute
python examples/example_oscillator.py    # after pip install -e .; FD errors around 1e-9 (about 1e-7 for dt: the 1e-3 s step)
```

If you change `docs/DBE-P.md`, check that its examples and the section numbers quoted in `dbep.py` and in the companion still match.
