# dbep

The dynamics back-end protocol (DBE-P) and its reference toolkit.

- `docs/DBE-P.md`: the protocol (draft v0.7). This is the source of truth.
- `docs/DBE-P-companion.md`: how didi becomes a back-end and how gogo consumes back-ends.
- `dbep.py`: types, `check`, the conformance suite, `Converted` (centre conversion wrapper), the quaternion helpers and `IAUOrientation`.
- `examples/example_oscillator.py`: a toy back-end and a consumer (run it: `python examples/example_oscillator.py`).
- `tests/test_dbep.py`: tests of the toolkit itself.

```bash
pip install -e ".[test]"
python -m pytest -q          # about 1 minute (JAX compilation)
```

didi implements the protocol; gogo uses it. Neither imports the other's internals. Both depend on this package.
