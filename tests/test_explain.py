# Tests for explain.py. Run them with: python -m pytest (in the exercise folder, with the virtual environment
# active). Each test calls one of your functions and checks what comes back. A failing test says what it called,
# what it expected and what came back.
# No test checks how important a column "should" be: the importance tests compare your table with scikit-learn's
# own permutation_importance, called with the same settings and the same seed, so the answers are exact. The other
# tests use small tables written here, with the lesson's numbers.
# You don't need to change this file.
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.inspection import permutation_importance
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

from bill_model import FEATURES, LABEL, split_homes
from explain import importance_table, plain_summary, top_features

# The lesson's table, rounded: the forest's permutation importance on the 140 test homes with seed 0.
LESSON_TABLE = pd.DataFrame({
    "feature": ["size_sqft", "insulation", "occupants", "heating", "has_pool", "year_built"],
    "importance": [16.31, 10.82, 4.15, 2.83, 1.67, -0.01],
    "spread": [1.01, 1.08, 0.54, 0.40, 0.33, 0.18],
})

EXAMPLE = [
    "This model estimates a home's monthly energy bill.",
    "On homes it hadn't seen, its estimates were off by about $11 a month on average.",
    "It leans most on the home's size, its insulation and how many people live there.",
    "That's what the model uses, not proof of what causes a bill.",
]


def check(call, got, expected):
    """Fails with a message that says what the call returned and what it should return."""
    assert got == expected, f"{call} should return {expected!r}, but it returned {got!r}."


def tree_model(X_train, y_train):
    """A small decision tree in the same kind of pipeline as bill_model's forest, quick to train for the tests."""
    prep = make_column_transformer(
        (OneHotEncoder(handle_unknown="ignore"), ["insulation", "heating"]),
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    return make_pipeline(prep, DecisionTreeRegressor(max_depth=6, random_state=0)).fit(X_train, y_train)


def expected_rows(model, X_test, y_test, seed):
    """What permutation_importance gives with the settings the task names, as {feature: (importance, spread)}."""
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=seed,
                                    scoring="neg_mean_absolute_error")
    return {feature: (mean, std) for feature, mean, std
            in zip(X_test.columns, result.importances_mean, result.importances_std)}


def table_rows(table):
    return {row.feature: (row.importance, row.spread) for row in table.itertuples()}


# importance_table

def test_1_importance_table_has_one_row_per_feature_and_the_columns_feature_importance_spread(homes):
    X_train, X_test, y_train, y_test = split_homes(homes, 42)
    table = importance_table(tree_model(X_train, y_train), X_test, y_test, 0)
    assert isinstance(table, pd.DataFrame), f"importance_table should return a DataFrame, but it returned {table!r}."
    check("the columns of importance_table(...)", list(table.columns), ["feature", "importance", "spread"])
    check("the features in importance_table(...), in any order", sorted(table["feature"]), sorted(FEATURES))


def test_2_importance_table_seed_0_matches_permutation_importance_with_10_repeats_and_the_mae(homes):
    X_train, X_test, y_train, y_test = split_homes(homes, 42)
    model = tree_model(X_train, y_train)
    check("each feature's (importance, spread) in importance_table(model, X_test, y_test, 0)",
          table_rows(importance_table(model, X_test, y_test, 0)), expected_rows(model, X_test, y_test, 0))


def test_3_importance_table_seed_7_matches_permutation_importance_with_random_state_7(homes):
    X_train, X_test, y_train, y_test = split_homes(homes, 42)
    model = tree_model(X_train, y_train)
    check("each feature's (importance, spread) in importance_table(model, X_test, y_test, 7)",
          table_rows(importance_table(model, X_test, y_test, 7)), expected_rows(model, X_test, y_test, 7))


def test_4_importance_table_puts_the_largest_importance_first(homes):
    X_train, X_test, y_train, y_test = split_homes(homes, 42)
    model = tree_model(X_train, y_train)
    table = importance_table(model, X_test, y_test, 0)
    expected = sorted(expected_rows(model, X_test, y_test, 0).items(), key=lambda item: (-item[1][0], item[0]))
    check("the order of the features in importance_table(...)", list(table["feature"]),
          [feature for feature, _ in expected])


def test_5_importance_table_ties_at_0_go_in_alphabetical_order(homes):
    # A model that keeps only the size column and drops the rest: shuffling any other column changes nothing,
    # so those five importances are all exactly 0.
    X_train, X_test, y_train, y_test = split_homes(homes, 42)
    size_only = make_column_transformer(("passthrough", ["size_sqft"]))
    model = make_pipeline(size_only, DecisionTreeRegressor(max_depth=6, random_state=0)).fit(X_train, y_train)
    table = importance_table(model, X_test, y_test, 0)
    check("the order of the features when five importances are all 0", list(table["feature"]),
          ["size_sqft", "has_pool", "heating", "insulation", "occupants", "year_built"])


def test_6_importance_table_numbers_the_rows_0_to_5_in_their_new_order(homes):
    X_train, X_test, y_train, y_test = split_homes(homes, 42)
    table = importance_table(tree_model(X_train, y_train), X_test, y_test, 0)
    check("the row numbers (index) of importance_table(...)", list(table.index), [0, 1, 2, 3, 4, 5])


# top_features

def test_7_top_features_3_gives_size_sqft_insulation_and_occupants():
    check("top_features(the lesson's table, 3)", top_features(LESSON_TABLE, 3),
          ["size_sqft", "insulation", "occupants"])


def test_8_top_features_1_gives_size_sqft():
    check("top_features(the lesson's table, 1)", top_features(LESSON_TABLE, 1), ["size_sqft"])


def test_9_top_features_10_gives_the_five_above_0_and_leaves_out_year_built():
    check("top_features(the lesson's table, 10)", top_features(LESSON_TABLE, 10),
          ["size_sqft", "insulation", "occupants", "heating", "has_pool"])


def test_10_top_features_leaves_out_an_importance_of_exactly_0():
    table = pd.DataFrame({"feature": ["size_sqft", "heating"], "importance": [5.0, 0.0], "spread": [0.5, 0.0]})
    check("top_features(size_sqft 5.0 and heating 0.0, 2)", top_features(table, 2), ["size_sqft"])


def test_11_top_features_returns_a_plain_list_of_names():
    got = top_features(LESSON_TABLE, 2)
    assert type(got) is list, f"top_features should return a list, but it returned {got!r}."
    assert all(type(name) is str for name in got), f"Each feature should be its name, a str, but got {got!r}."


# plain_summary

def test_12_plain_summary_the_example_size_insulation_occupants_and_11_27():
    check('plain_summary(["size_sqft", "insulation", "occupants"], 11.27)',
          plain_summary(["size_sqft", "insulation", "occupants"], 11.27), EXAMPLE)


def test_13_plain_summary_one_feature_has_pool_and_8_6_rounds_to_9():
    got = plain_summary(["has_pool"], 8.6)
    check('plain_summary(["has_pool"], 8.6)[1]', got[1],
          "On homes it hadn't seen, its estimates were off by about $9 a month on average.")
    check('plain_summary(["has_pool"], 8.6)[2]', got[2], "It leans most on whether it has a pool.")


def test_14_plain_summary_two_features_insulation_and_heating_and_12_4_rounds_to_12():
    got = plain_summary(["insulation", "heating"], 12.4)
    check('plain_summary(["insulation", "heating"], 12.4)[1]', got[1],
          "On homes it hadn't seen, its estimates were off by about $12 a month on average.")
    check('plain_summary(["insulation", "heating"], 12.4)[2]', got[2],
          "It leans most on its insulation and its heating.")


def test_15_plain_summary_no_features_says_it_doesnt_lean_clearly_on_any():
    got = plain_summary([], 15.0)
    check("plain_summary([], 15.0)[2]", got[2], "It doesn't lean clearly on any one feature.")


def test_16_plain_summary_returns_four_sentences_in_a_list():
    got = plain_summary(["year_built"], 20.2)
    assert type(got) is list, f"plain_summary should return a list, but it returned {got!r}."
    check("how many sentences plain_summary(...) returns", len(got), 4)
    check("plain_summary(...)[0]", got[0], "This model estimates a home's monthly energy bill.")
    check("plain_summary(...)[3]", got[3], "That's what the model uses, not proof of what causes a bill.")
