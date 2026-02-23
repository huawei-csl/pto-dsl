# Tests required for new pull requests

Locally verify [./tests](./tests) and [./examples](./examples) on NPU device:

```bash
pytest -v tests
python examples/validate_all_examples.py
```

For non-device tests such as frontend and IR, cover them by [.github/workflows/ci.yml](./.github/workflows/ci.yml)
