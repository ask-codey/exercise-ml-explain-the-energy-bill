"""Greenway Home Energy: the model to explain. Ready to use; you don't need to change this file.

homes is a pandas table (a DataFrame) with one row per home and the columns home_id,
size_sqft, year_built, insulation, heating, occupants, has_pool and monthly_bill, like
the one pd.read_csv("home_energy.csv") gives you.
"""
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

# The columns the model looks at (its features), in this order.
FEATURES = ["size_sqft", "year_built", "insulation", "heating", "occupants", "has_pool"]

# The column the model learns to guess (its label).
LABEL = "monthly_bill"


def split_homes(homes, seed):
    """Split the homes into training and test parts: X_train, X_test, y_train, y_test.

    A fifth of the homes are held back for testing, and seed is passed on as
    random_state. With the 700 homes in home_energy.csv: 560 training homes and
    140 test homes.
    """
    return train_test_split(homes[FEATURES], homes[LABEL], test_size=0.2, random_state=seed)


def build_bill_model(seed):
    """Return an unfitted pipeline: one-hot encode insulation and heating, keep the
    other columns as they are, then a random forest of 200 trees with seed as its
    random_state. Call fit on it with the training homes before you use it.
    """
    prep = make_column_transformer(
        (OneHotEncoder(handle_unknown="ignore"), ["insulation", "heating"]),
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    return make_pipeline(prep, RandomForestRegressor(n_estimators=200, random_state=seed))
