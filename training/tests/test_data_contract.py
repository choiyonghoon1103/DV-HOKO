from pathlib import Path

import numpy as np
from scipy.io import savemat

from hoko.data.hust import discover_hust_records


def test_hust_adapter_enumerates_complete_cells(tmp_path: Path):
    for suffix in ("5", "6", "7", "8"):
        for label in ("N", "I", "O", "B"):
            for load in ("00", "02", "04"):
                savemat(
                    tmp_path / f"{label}{suffix}{load}.mat",
                    {"data": np.zeros(51_200, dtype=np.float32), "fs": 25.0},
                )
    rows = discover_hust_records(tmp_path)
    assert len(rows) == 48
    assert {(row["bearing"], row["class_index"]) for row in rows} == {
        (bearing, label) for bearing in ("6205", "6206", "6207", "6208") for label in range(4)
    }

