# Instellution

**1. Clone the repository:**

```bash
git clone https://github.com/bryceshirley/ptychobench-workexperience.git
cd ptychobench-workexperience
```

**2. Install `uv` (Package Manager)** via instructions [here](https://docs.astral.sh/uv/getting-started/installation/).

**3. Install dependencies:**
```bash
uv sync
```

## Building the Documentation Locally

Run the following command to build the documentation locally and serve it on your machine:

```bash
uv run mkdocs serve
```

## Running Unit Tests

Testing is the foundation of reliable research. By running the test suite, you verify that your physical approximations, operator logic, and grid calculations are producing the correct numerical output.

We use `pytest` to manage our testing suite. Because we are using `uv`, you can run the entire suite directly from your terminal with the following command:

```bash
uv run pytest
```

### Common Commands

* **Run all tests:**

```bash
uv run pytest
```


* **Run tests with verbose output (shows individual test names):**
```bash
uv run pytest -v
```


* **Run a specific test file (e.g., testing your operators):**
```bash
uv run pytest tests/test_operators.py

```

## What do the results mean?

* **Green (dots):** All tests passed.
* **Red (F):** A test failed.