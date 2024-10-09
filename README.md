# MLP language modelling in JAX

## Build Environment

1. install [uv](https://github.com/astral-sh/uv)

```bash
# On macOS and Linux.
$ curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows.
$ powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# With pip.
$ pip install uv
```

2. create virtual enviroment

```
uv sync
```

3. activate pre-commit

```
uv run pre-commit install
```
Note that this environment is for TPU. Adjust accordingly for your device.
