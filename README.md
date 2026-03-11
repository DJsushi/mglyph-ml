# Malleable Glyph Machine Learning

## Run it yourself

First of all, install `uv`. It can be installed as a system binary, or as a pip package inside a local Python environment. The tutorial is available [here](https://docs.astral.sh/uv/getting-started/installation/).

Then, `cd` into the project's root directory and run `uv sync`. This will create a `.venv` inside the project's root directory containing the virtual environment, while ensuring that the packages inside it match exactly the packages specified in the `pyproject.toml` file.

Then, depending on your use case, you can either:
- Run `uv run python my_python_file.py` to run the file under this virtual environment,
- or simply select your Python virtual environment inside your Jupyter notebook to run the notebook using this project's venv.

