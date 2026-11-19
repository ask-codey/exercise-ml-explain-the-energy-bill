"""Makes home_energy.csv: 700 made-up homes and their monthly energy bills, for Greenway Home Energy.

Ready to use; you don't need to run it. The CSV it made is already in this folder,
and the lessons, the tests and report.py all read that file.

Every number in the file is made up, and the pattern is built in on purpose, so you
can check what a model leans on against what was really planted. A real dataset's
pattern is what you're trying to find; this one is planted:

    monthly_bill = 35
                   + 0.045 for each square foot (size_sqft)
                   + 40 for poor insulation, or 15 for fair (0 for good)
                   + 25 for electric heat, or 10 for gas (0 for a heat pump)
                   + 8 for each person living there (occupants)
                   + 30 for a pool (has_pool)
                   + a random wobble (a standard deviation of 12 dollars)
    then rounded to the cent.

year_built has no effect of its own on the bill. But newer homes were built with
better insulation, so year_built and insulation move together, and a model can use
one to stand in for the other.

The random numbers come from a fixed seed, 502, so this script makes the same file
every time on the same version of NumPy. NumPy doesn't promise the same random
numbers from a seed across its versions (NEP 19), so the CSV in this folder, not
this script, is what keeps every learner's numbers the same.
"""
import numpy as np
import pandas as pd

SEED = 502
HOMES = 700


def make_homes(seed=SEED, count=HOMES):
    """Return the made-up homes as a DataFrame, one row per home."""
    rng = np.random.default_rng(seed)
    size_sqft = (rng.normal(1900, 550, size=count).clip(700, 4200) / 10).round() * 10
    year_built = rng.integers(1950, 2025, size=count)

    # Newer homes are likelier to have good insulation: a home's chance moves with its year.
    newness = (year_built - 1950) / 74
    draw = rng.random(count)
    poor_share = 0.75 - 0.70 * newness
    fair_share = 0.20 + 0.10 * np.sin(np.pi * newness)
    insulation = np.where(draw < poor_share, "poor",
                          np.where(draw < poor_share + fair_share, "fair", "good"))

    heating = rng.choice(["gas", "electric", "heat pump"], size=count, p=[0.45, 0.30, 0.25])
    occupants = rng.choice([1, 2, 3, 4, 5, 6], size=count, p=[0.15, 0.30, 0.20, 0.20, 0.10, 0.05])
    has_pool = (rng.random(count) < 0.12).astype(int)
    wobble = rng.normal(0, 12, size=count)

    bill = (35 + 0.045 * size_sqft
            + np.where(insulation == "poor", 40, np.where(insulation == "fair", 15, 0))
            + np.where(heating == "electric", 25, np.where(heating == "gas", 10, 0))
            + 8 * occupants + 30 * has_pool + wobble)

    return pd.DataFrame({
        "home_id": [f"GH-{1001 + i}" for i in range(count)],
        "size_sqft": size_sqft.astype(int),
        "year_built": year_built.astype(int),
        "insulation": insulation,
        "heating": heating,
        "occupants": occupants.astype(int),
        "has_pool": has_pool,
        "monthly_bill": bill.round(2),
    })


if __name__ == "__main__":
    homes = make_homes()
    homes.to_csv("home_energy.csv", index=False)
    print("Wrote home_energy.csv:", len(homes), "homes.")
