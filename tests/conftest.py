import pandas as pd
import pytest


@pytest.fixture
def transactions():
    return pd.DataFrame(
        {
            "Time": [10, 20, 30, 40, 50, 60],
            "Amount": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "Class": [0, 0, 0, 0, 1, 1],
            "V1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        }
    )
