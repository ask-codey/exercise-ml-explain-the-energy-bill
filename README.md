# Exercise: Explain the Energy Bill

Exercise code for the lesson **Exercise: Explain the Energy Bill** in the Machine Learning with Python course on [ask-codey.com](https://www.ask-codey.com).

Greenway Home Energy, a made-up utility, estimates next month's energy bill for 700 homes with a random forest, and its customers want to know why. You'll write three functions in `explain.py` that measure what the model leans on with permutation importance, pick the few features that matter most, and turn them into sentences a homeowner can read. Tests call each function and check what it returns.

## What's here

- `home_energy.csv`: the 700 homes, our own made-up data: each home's `home_id`, `size_sqft`, `year_built`, `insulation` (poor, fair or good), `heating` (gas, electric or heat pump), `occupants`, `has_pool` (1 or 0) and `monthly_bill`. The lessons, the tests and `report.py` all read this file. Ready to use.
- `seed_data.py`: the script that made `home_energy.csv`, so you can see how the data was built and the pattern planted in it. You don't need to run it: NumPy doesn't promise the same random numbers from a seed on every version, so the CSV, not the script, is what keeps everyone's numbers the same. Ready to use.
- `bill_model.py`: the model to explain: `FEATURES`, `LABEL`, `split_homes(homes, seed)` (560 training homes and 140 test homes) and `build_bill_model(seed)` (one-hot encoding and a random forest of 200 trees, in a pipeline). Ready to use.
- `explain.py`: the three functions, `importance_table`, `top_features` and `plain_summary`, with `NAMES`, each column's name in a homeowner's words. Each function stops with `NotImplementedError: Not built yet.` for now. The docstring under each says what it should do, with examples. For you to finish.
- `explore.ipynb`: a notebook for looking at the homes and trying your functions as you write them. The tests don't read it. Ready to use, and yours to change.
- `report.py`: trains the forest, measures it on the test homes and explains it with your three functions, printing what each step gave. Ready to use.
- `tests/test_explain.py`: tests that call your functions and check what comes back. You don't need to change this file.
- `tests/conftest.py`: gives each test a fresh copy of the 700 homes from `home_energy.csv`. Ready to use.
- `requirements.txt`: the packages this exercise needs, each at an exact version: scikit-learn, pandas, NumPy, matplotlib (for drawing partial dependence in the notebook), ipykernel (so VS Code can run the notebook) and pytest. Ready to use.
- `pytest.ini`: tells pytest where the tests are and where to find `explain.py` and `bill_model.py`. Ready to use.
- `.gitignore`: tells Git to leave out the virtual environment and the files Python, pytest and the notebook make as they run. Ready to use.

## Before you start

The packages go in a virtual environment: a private set of packages for this folder only. In VS Code's terminal, in this folder, run these three commands.

On Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On a Mac (zsh):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

- The first command makes the virtual environment, in a folder named `.venv`.
- The second turns it on for this terminal: the prompt starts with `(.venv)` from then on.
- The third installs the packages `requirements.txt` lists into it.

*Virtual Environments and pip* explains all three. The `.venv` folder is in `.gitignore`, so Git leaves it out: these commands make it again on any computer.

If PowerShell answers that running scripts is disabled on this system, run this once, then run the second command again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Each new terminal starts with the virtual environment off. If the prompt doesn't start with `(.venv)`, run the second command again. (VS Code often turns it on for you.)

To look at the homes, open `explore.ipynb`, choose the `.venv` kernel with **Select Kernel** (top right), and run the cells.

## Check your work

```bash
python -m pytest
```

At first every test fails: that is expected, because each function stops with a `NotImplementedError` for now. Each function you finish turns its tests green. A failing test's message says what it called, what it expected and what came back, for example *top_features(the lesson's table, 3) should return ['size_sqft', 'insulation', 'occupants'], but it returned ['size_sqft', 'insulation'].*

No test checks how important a column "should" be. The importance tests compare your table with scikit-learn's own `permutation_importance`, called with the same settings and the same seed, so their answers are exact; the other tests use small tables with the lesson's numbers.

To see your functions explain the real forest, run:

```bash
python report.py
```

It takes a little while: the forest has 200 trees, and each column is shuffled 10 times.

## The rules

- Keep each function's name and parameters as they are, and keep `NAMES` as it is.
- Measure importance on the test homes the function is given, never on training homes.
- Pass `seed` on as `random_state`, so the same seed always gives the same table.
- Return each answer; don't only print it.
- Don't change the tests, `conftest.py`, `home_energy.csv`, `seed_data.py`, `bill_model.py` or `report.py`.

## Tips

- Try each step in `explore.ipynb` first, on the real homes, then move the lines that work into `explain.py`.
- `permutation_importance` gives back `importances_mean` and `importances_std`, one number per column, in the order of `X_test`'s columns.
- `sort_values` can sort by two columns at once: give it a list of column names and a list of `ascending` settings, one for each.
- A table's rows can be picked with a condition, such as `table[table["importance"] > 0]`.
- After you change `explain.py`, restart the notebook's kernel before you try it there, so the notebook imports your newest code.

## Hand it in

When it works, push your solution to a public repo of your own on GitHub, then paste that repo's address into **Submit your exercise** on the lesson page at [ask-codey.com](https://www.ask-codey.com). That's what completes the lesson. The lesson page shows the Git commands, step by step.
