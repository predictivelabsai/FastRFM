"""Driver features on a hand-built slice of the rel-f1 schema. No download."""

import numpy as np
import pandas as pd

from fastrfm.relbench import driver_matrix


def _tables(status_future: int, status_past: int) -> dict[str, pd.DataFrame]:
    results = pd.DataFrame(
        {
            "resultId": [1, 2],
            "raceId": [1, 2],
            "driverId": [7, 7],
            "constructorId": [3, 9],
            "statusId": [status_past, status_future],
            "positionOrder": [2.0, 10.0],
            "points": [18.0, 0.0],
            "date": [pd.Timestamp("2004-06-01"), pd.Timestamp("2004-08-01")],
        }
    )
    # A teammate's result, so the constructor aggregate is not just the driver.
    teammate = pd.DataFrame(
        {
            "resultId": [3],
            "raceId": [1],
            "driverId": [8],
            "constructorId": [3],
            "statusId": [5],
            "positionOrder": [14.0],
            "points": [0.0],
            "date": [pd.Timestamp("2004-06-01")],
        }
    )
    results = pd.concat([results, teammate], ignore_index=True)
    return {"results": results}


def test_future_race_is_invisible_and_the_past_is_not():
    anchor = np.array([pd.Timestamp("2004-07-01")])
    drivers = np.array([7])
    past = driver_matrix(_tables(status_future=11, status_past=1), drivers, anchor, mode="rfm")
    future_flipped = driver_matrix(_tables(status_future=1, status_past=1), drivers, anchor, mode="rfm")
    assert np.allclose(past, future_flipped)
    past_flipped = driver_matrix(_tables(status_future=11, status_past=4), drivers, anchor, mode="rfm")
    assert not np.allclose(past, past_flipped)


def test_flat_ignores_the_teammate():
    anchor = np.array([pd.Timestamp("2004-07-01")])
    drivers = np.array([7])
    flat = driver_matrix(_tables(status_future=11, status_past=1), drivers, anchor, mode="flat")
    # One past race, finished (status 1), so the driver's own DNF rate is 0.
    # The teammate's DNF is a join, and flat does not get a column for it.
    assert flat.shape == (1, 5)
    assert flat[0, 0] == 1  # n_races
    assert flat[0, 1] == 0  # dnf rate
    relational = driver_matrix(_tables(status_future=11, status_past=1), drivers, anchor, mode="rfm")
    assert relational.shape == (1, 10)
    # Qualifying is absent, so that slot stays 0; constructor peer DNF rate is 1
    # (the teammate did not finish) and lives in extra column 1.
    assert relational[0, 6] == 1
