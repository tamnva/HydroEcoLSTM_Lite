Deployment Guide

This document describes common deployment tasks: building documentation, packaging the Python package, and publishing releases.

Build documentation

MkDocs (recommended):

```powershell
# install dependencies
pip install -r docs/requirements-mkdocs.txt
# serve locally
mkdocs serve
# build static site
mkdocs build
```

Sphinx:

```powershell
pip install -r docs_sphinx/requirements-sphinx.txt
sphinx-build -b html docs_sphinx build/html
```

Packaging and publishing

Create source and wheel distributions and upload to PyPI using `twine`:

```powershell
# build
pip install build twine
python -m build
# check artifacts in dist/
# upload to TestPyPI first (recommended)
python -m twine upload --repository testpypi dist/*
# when ready, upload to PyPI
python -m twine upload dist/*
```

GitHub release workflow

- Create a release tag (semantic versioning recommended):

```powershell
git tag -a v0.1.0 -m "Release v0.1.0"
git push origin v0.1.0
```

- Draft a GitHub release and attach built artifacts if desired.

Conda / environment

A conda environment YAML is provided at `environment/environment.yml`. To create an environment:

```powershell
conda env create -f environment/environment.yml
conda activate <env-name>
```

Caveats and recommendations

- Prefer publishing to TestPyPI for validation before uploading to PyPI.
- Ensure sensitive secrets (API keys, PyPI tokens) are stored as GitHub Actions secrets and never committed to the repo.
- Automate docs and releases via GitHub Actions for repeatability.

GitHub Actions notes

- A workflow `build-docs.yml` is included to build and deploy the MkDocs site to `gh-pages`.
- A workflow `build-sphinx.yml` is included to build Sphinx HTML and upload artifacts for inspection on PRs and pushes.
- A workflow `publish-pypi.yml` is included to publish package releases when you push a tag like `v0.1.0`.

Set these repository secrets in `Settings -> Secrets -> Actions`:

- `PYPI_API_TOKEN`: API token for PyPI (upload to production PyPI)
- `TEST_PYPI_API_TOKEN`: API token for TestPyPI (optional, used if `PYPI_API_TOKEN` is not set)

When both tokens are set the workflow will prefer `PYPI_API_TOKEN` (production). If only the TestPyPI token is set, releases will be uploaded to TestPyPI.
