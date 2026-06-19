Contributing to HydroEcoLSTM-Lite

Thanks for your interest in contributing! This short guide explains how to report issues, contribute code, and prepare changes for review.

How to report bugs

- Search existing issues to avoid duplicates.
- Create a new issue with a descriptive title and a minimal reproduction (config, short dataset or steps).
- Include environment details: Python version, OS, package versions (from `pip freeze`).

Contributing code

1. Fork the repository and create a feature branch from `main`:

```bash
git checkout -b feature/my-change
```

2. Follow style and testing guidelines:
- Write clear docstrings for modules, classes and public functions (NumPy or Google style preferred).
- Keep API compatibility in mind; prefer small, focused changes.
- Add or update unit tests where appropriate.

3. Run formatting and tests locally:

```bash
# optional: create virtual env
pip install -e .
# run tests (if present)
pytest
```

4. Commit and push your branch using descriptive commit messages:

- Use present-tense short summary, e.g. "Fix scaler handling for constant columns"
- Reference issues when appropriate: "Fixes #123"

5. Open a Pull Request against `main` and provide a clear description of the change, why it is needed, and any relevant notes for reviewers.

Code review and merge

- PRs are reviewed by maintainers. Expect feedback and be responsive to requested changes.
- Maintain a clean history if requested (squash or rebase when suggested).

Community and conduct

- Be respectful and constructive. If you plan a larger change, open an issue or discussion first to agree on the approach.

Questions or help

Open an issue or contact the maintainers listed in the repository metadata.
