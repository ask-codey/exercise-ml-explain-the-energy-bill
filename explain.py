"""Greenway Home Energy: three functions that explain a model, for you to finish.

The docstring under each function says what it should do, with examples.
Replace each raise NotImplementedError("Not built yet.") line with your own code.
Return each answer: the tests read what comes back, not what's printed.

The imports below are everything the three functions need.
"""
import pandas as pd
from sklearn.inspection import permutation_importance

# Each column's name in words a homeowner uses. Ready to use.
NAMES = {
    "size_sqft": "the home's size",
    "year_built": "the year it was built",
    "insulation": "its insulation",
    "heating": "its heating",
    "occupants": "how many people live there",
    "has_pool": "whether it has a pool",
}


def importance_table(model, X_test, y_test, seed):
    """Return the model's permutation importance on the test homes, as a sorted table.

    Call permutation_importance with the fitted model, X_test and y_test, and these
    settings: n_repeats=10, random_state=seed and scoring="neg_mean_absolute_error",
    so each importance is in dollars added to the average miss.

    Return a DataFrame with one row per column of X_test and three columns:
    feature (the column's name), importance (importances_mean) and spread
    (importances_std), not rounded. Sort it largest importance first; when two
    importances are equal, the features go in alphabetical order. Number the rows
    0, 1, 2 and so on in their new order.

    With the lesson's forest on the 140 test homes and seed 0, the first rows are:
       feature  importance  spread
     size_sqft       16.31    1.01
    insulation       10.82    1.08
    (rounded here to two decimal places)
    """
    raise NotImplementedError("Not built yet.")


def top_features(table, n):
    """Return the names of the first n features in a sorted importance table, as a list.

    Leave out every feature whose importance is 0 or less: shuffling it didn't make
    the guesses worse, so it isn't something the model leans on. If fewer than n
    features are left, return them all.

    With the lesson's table (size_sqft 16.31, insulation 10.82, occupants 4.15,
    heating 2.83, has_pool 1.67, year_built -0.01):
    top_features(table, 3)  -> ["size_sqft", "insulation", "occupants"]
    top_features(table, 10) -> ["size_sqft", "insulation", "occupants", "heating", "has_pool"]
    """
    raise NotImplementedError("Not built yet.")


def plain_summary(top, mae):
    """Return four sentences, as a list of strings, that explain the model to a homeowner.

    top is a list of feature names, as top_features gives them; mae is the model's
    average miss on the test homes, in dollars. Use NAMES for each feature's words,
    and round mae to whole dollars with round().

    1. This model estimates a home's monthly energy bill.
    2. On homes it hadn't seen, its estimates were off by about $<mae> a month on average.
    3. It leans most on <the names>.
       One name: "It leans most on its insulation."
       Two: "It leans most on its insulation and its heating."
       Three or more: commas between them and "and" before the last:
       "It leans most on the home's size, its insulation and how many people live there."
       No names at all: "It doesn't lean clearly on any one feature."
    4. That's what the model uses, not proof of what causes a bill.

    plain_summary(["size_sqft", "insulation", "occupants"], 11.27) ->
    ["This model estimates a home's monthly energy bill.",
     "On homes it hadn't seen, its estimates were off by about $11 a month on average.",
     "It leans most on the home's size, its insulation and how many people live there.",
     "That's what the model uses, not proof of what causes a bill."]
    """
    raise NotImplementedError("Not built yet.")
