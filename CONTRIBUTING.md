# Contributing to AI-Driven Road Safety System

Contributions, bug reports, and pull requests are welcome!

## Setting Up for Development

1. Fork and clone the repository:
   ```bash
   git clone https://github.com/TejaPriyan/AI-DRIVEN-ROAD-SAFETY-SYSTEM-HELMET-DETECTION-AND-NUMBERPLATERECOGNITION-USING-YOLOV8.git
   cd AI-DRIVEN-ROAD-SAFETY-SYSTEM-HELMET-DETECTION-AND-NUMBERPLATERECOGNITION-USING-YOLOV8
   ```
2. Install runtime dependencies (or test dependencies for unit checks):
   ```bash
   pip install -r requirements.txt
   pip install pytest
   ```

## Running Checks Before Submitting a PR

- Verify Python syntax across all CLI entrypoints:
  ```bash
  python -m py_compile detect.py predict.py train.py val.py
  ```
- Run the unit test suite:
  ```bash
  pytest -v
  ```

## Submitting Changes

1. Create a feature or bugfix branch from `main`.
2. Keep changes focused and ensure all existing and new tests pass.
3. Strip output cells from any modified Jupyter notebooks (`.ipynb`) before committing.
4. Open a pull request against `main` describing the problem solved and how it was verified.

## License

By contributing, you agree that your contributions will be licensed under the [GNU Affero General Public License v3.0 (AGPL-3.0)](LICENSE).
