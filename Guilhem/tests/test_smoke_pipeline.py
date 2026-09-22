from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _write_patch(path: Path, value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.full((4, 64, 64), fill_value=value, dtype=np.float32)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=64,
        width=64,
        count=4,
        dtype="float32",
        transform=from_origin(0, 0, 1, 1),
    ) as dst:
        for i in range(4):
            dst.write(arr[i], i + 1)


class SmokePipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp(prefix="sdm-smoke-"))
        self.data_root = self.tmpdir / "data"
        self.artifacts = self.tmpdir / "artifacts"
        self.cache = self.tmpdir / ".cache"

        self._build_dataset()
        self.config_path = self._write_config()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _build_dataset(self) -> None:
        self.data_root.mkdir(parents=True, exist_ok=True)

        pa_rows = [
            {"surveyId": 1, "spId": 101, "lat": 45.0, "lon": 3.0, "presence": 1},
            {"surveyId": 1, "spId": 102, "lat": 45.0, "lon": 3.0, "presence": 1},
            {"surveyId": 2, "spId": 101, "lat": 45.2, "lon": 3.1, "presence": 1},
            {"surveyId": 3, "spId": 103, "lat": 46.0, "lon": 4.0, "presence": 1},
            {"surveyId": 4, "spId": 102, "lat": 46.1, "lon": 4.1, "presence": 1},
        ]
        po_rows = [
            {"surveyId": 5, "spId": 101, "lat": 45.5, "lon": 3.5, "presence": 1},
            {"surveyId": 6, "spId": 103, "lat": 46.2, "lon": 4.2, "presence": 1},
        ]
        test_rows = [
            {"surveyId": 10, "lat": 45.1, "lon": 3.2},
            {"surveyId": 11, "lat": 46.3, "lon": 4.3},
        ]

        pd.DataFrame(pa_rows).to_csv(self.data_root / "GLC25_PA_metadata_train.csv", index=False)
        pd.DataFrame(po_rows).to_csv(self.data_root / "GLC25_PO_metadata_train.csv", index=False)
        pd.DataFrame(test_rows).to_csv(self.data_root / "GLC25_PA_metadata_test.csv", index=False)

        # Satellite patches
        for sid in [1, 2, 3, 4]:
            _write_patch(self.data_root / "SatelitePatches" / "PA-train" / f"{sid}.tiff", value=float(sid))
        for sid in [5, 6]:
            _write_patch(self.data_root / "SatelitePatches" / "PO-train" / f"{sid}.tiff", value=float(sid))
        for sid in [10, 11]:
            _write_patch(self.data_root / "SatelitePatches" / "PA-test" / f"{sid}.tiff", value=float(sid))

        # Landsat values
        for split, ids in [("PA-train", [1, 2, 3, 4]), ("PA-test", [10, 11]), ("PO-train", [5, 6])]:
            landsat_dir = self.data_root / "SateliteTimeSeries-Landsat" / "values" / split
            landsat_dir.mkdir(parents=True, exist_ok=True)
            for band in ["band_r", "band_nir"]:
                rows = []
                for sid in ids:
                    rows.append({"surveyId": sid, "t1": sid * 0.1, "t2": sid * 0.2, "t3": sid * 0.3, "t4": sid * 0.4})
                pd.DataFrame(rows).to_csv(landsat_dir / f"{band}.csv", index=False)

        # Bioclim monthly
        bc_dir = self.data_root / "BioclimTimeSeries" / "values"
        bc_dir.mkdir(parents=True, exist_ok=True)
        for name, ids in [
            ("GLC25-PA-train-bioclimatic_monthly.csv", [1, 2, 3, 4]),
            ("GLC25-PA-test-bioclimatic_monthly.csv", [10, 11]),
        ]:
            rows = []
            for sid in ids:
                row = {"surveyId": sid}
                for i in range(8):
                    row[f"m{i}"] = sid * (i + 1) * 0.01
                rows.append(row)
            pd.DataFrame(rows).to_csv(bc_dir / name, index=False)

        # Environmental values
        env_dir = self.data_root / "EnvironmentalValues" / "Elevation"
        env_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [{"surveyId": sid, "elev": sid * 10.0} for sid in [1, 2, 3, 4]]
        ).to_csv(env_dir / "GLC25-PA-train-elevation.csv", index=False)
        pd.DataFrame(
            [{"surveyId": sid, "elev": sid * 10.0} for sid in [10, 11]]
        ).to_csv(env_dir / "GLC25-PA-test-elevation.csv", index=False)
        pd.DataFrame(
            [{"surveyId": sid, "elev": sid * 10.0} for sid in [5, 6]]
        ).to_csv(env_dir / "GLC25-PO-train-elevation.csv", index=False)

    def _write_config(self) -> Path:
        cfg = {
            "data": {
                "cache_dir": str(self.cache),
                "num_workers": 0,
                "spatial_grid_size": 0.5,
                "min_species_per_sample": 1,
                "max_species_per_sample": 3,
            },
            "training": {
                "batch_size": 2,
                "epochs": 1,
                "lr": 5e-4,
                "patience": 1,
                "folds": 2,
                "amp": False,
                "max_train_samples": 4,
                "max_val_samples": 2,
            },
            "pseudo_label": {
                "batch_size": 2,
                "t_pos": 0.3,
                "t_neg": 0.1,
                "max_negatives_per_sample": 3,
            },
            "output": {"log_interval": 1},
        }
        config_path = self.tmpdir / "config.yaml"
        with config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(cfg, handle)
        return config_path

    def _run(self, args: list[str]) -> None:
        subprocess.run(
            [sys.executable] + args,
            cwd=ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

    def test_end_to_end_cli(self) -> None:
        teacher_dir = self.artifacts / "teacher"
        student_dir = self.artifacts / "student"
        pseudo_path = self.artifacts / "pseudo" / "pseudo_labels.pt"
        submission_path = self.artifacts / "submission.csv"

        self._run(
            [
                "scripts/train_pa.py",
                "--data-root",
                str(self.data_root),
                "--config",
                str(self.config_path),
                "--fold",
                "0",
                "--output-dir",
                str(teacher_dir),
            ]
        )

        teacher_ckpt = teacher_dir / "teacher_fold0.pt"
        self.assertTrue(teacher_ckpt.exists())

        self._run(
            [
                "scripts/pseudo_label_po.py",
                "--teacher-checkpoint",
                str(teacher_ckpt),
                "--data-root",
                str(self.data_root),
                "--output",
                str(pseudo_path),
            ]
        )
        self.assertTrue(pseudo_path.exists())

        self._run(
            [
                "scripts/train_student.py",
                "--teacher-checkpoint",
                str(teacher_ckpt),
                "--pseudo-labels",
                str(pseudo_path),
                "--data-root",
                str(self.data_root),
                "--output-dir",
                str(student_dir),
            ]
        )

        student_ckpt = student_dir / "student_fold0.pt"
        self.assertTrue(student_ckpt.exists())

        self._run(
            [
                "scripts/predict_test.py",
                "--checkpoint",
                str(student_ckpt),
                "--data-root",
                str(self.data_root),
                "--output-csv",
                str(submission_path),
            ]
        )
        self.assertTrue(submission_path.exists())

        self._run(["scripts/validate_submission.py", "--submission", str(submission_path)])


if __name__ == "__main__":
    unittest.main()
