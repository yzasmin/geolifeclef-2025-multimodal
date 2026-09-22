from __future__ import annotations

import argparse
import unittest
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from sdm.calibration import evaluate_calibration_grid
from sdm.imbalance import build_pa_po_sample_weights
from sdm.runtime_overrides import resolve_int_with_source
from scripts.long_run_script import _make_steps


@dataclass
class DummyPseudoRecord:
    survey_id: int
    positive_idx: list[int]
    negative_idx: list[int]


class CalibrationAndSamplerTest(unittest.TestCase):
    def test_resolve_int_with_source_priority(self) -> None:
        v, src = resolve_int_with_source(cli_value=256, ckpt_value=8, default_value=64)
        self.assertEqual(v, 256)
        self.assertEqual(src, "cli")

        v, src = resolve_int_with_source(cli_value=None, ckpt_value=8, default_value=64)
        self.assertEqual(v, 8)
        self.assertEqual(src, "checkpoint")

        v, src = resolve_int_with_source(cli_value=None, ckpt_value=None, default_value=64)
        self.assertEqual(v, 64)
        self.assertEqual(src, "default")

    def test_calibration_grid_prefers_lower_alpha_when_set_size_overpredicts(self) -> None:
        y_true = np.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        probs = torch.tensor(
            [
                [0.95, 0.40, 0.10],
                [0.95, 0.90, 0.20],
                [0.20, 0.88, 0.10],
            ],
            dtype=torch.float32,
        )
        set_sizes = torch.tensor([3.2, 3.1, 2.8], dtype=torch.float32)

        result = evaluate_calibration_grid(
            y_true=y_true,
            probs=probs,
            set_sizes=set_sizes,
            min_k=1,
            alphas=[0.55, 1.0],
            max_k_grid=[2, 3],
        )

        self.assertAlmostEqual(result.alpha, 0.55)
        self.assertEqual(result.max_k, 2)
        self.assertGreater(result.metrics["f1_samples"], 0.75)

    def test_weighted_sampler_builder_outputs_valid_ratio(self) -> None:
        pa_ids = [1, 2, 3, 4]
        po_ids = [10, 11]

        pa_labels = {
            1: [101],
            2: [101, 102],
            3: [103],
            4: [104],
        }
        pseudo_records = {
            10: DummyPseudoRecord(survey_id=10, positive_idx=[0], negative_idx=[1]),
            11: DummyPseudoRecord(survey_id=11, positive_idx=[2], negative_idx=[0]),
        }
        species_to_index = {101: 0, 102: 1, 103: 2, 104: 3}

        pa_df = pd.DataFrame(
            {
                "surveyId": pa_ids,
                "lat": [45.0, 45.2, 46.1, 46.0],
                "lon": [3.0, 3.1, 4.0, 4.1],
            }
        )
        po_df = pd.DataFrame(
            {
                "surveyId": po_ids,
                "lat": [45.1, 46.2],
                "lon": [3.1, 4.1],
            }
        )

        out = build_pa_po_sample_weights(
            pa_train_ids=pa_ids,
            po_train_ids=po_ids,
            pa_labels=pa_labels,
            pseudo_records=pseudo_records,
            species_to_index=species_to_index,
            pa_survey_df=pa_df,
            pa_survey_id_col="surveyId",
            pa_lat_col="lat",
            pa_lon_col="lon",
            po_survey_df=po_df,
            po_survey_id_col="surveyId",
            po_lat_col="lat",
            po_lon_col="lon",
            grid_size=1.0,
            target_pa_ratio=2.0,
            min_weight=0.2,
            max_weight=5.0,
        )

        self.assertEqual(len(out.pa_weights), len(pa_ids))
        self.assertEqual(len(out.po_weights), len(po_ids))
        self.assertTrue(np.all(out.pa_weights >= 0.0))
        self.assertTrue(np.all(out.po_weights >= 0.0))

        ratio = out.report["observed_weight_mass_ratio_pa_to_po"]
        self.assertGreater(ratio, 1.2)
        self.assertLess(ratio, 3.0)

    def test_long_run_steps_include_all_folds_and_final_stages(self) -> None:
        args = argparse.Namespace(
            data_root="data",
            config="/tmp/cfg.yaml",
            output_root="/tmp/long_run",
            folds=[0, 1],
            gpu_id=0,
            pseudo_batch_size=256,
            pseudo_num_workers=2,
            max_retries=1,
            resume=True,
            stop_on_error=True,
            python_bin="python3",
        )
        steps, student_ckpts, calib_path, submission_csv = _make_steps(args)

        self.assertEqual(len(student_ckpts), 2)
        self.assertEqual(steps[0].name, "audit_modalities")
        self.assertEqual(steps[1].name, "train_pa_fold0")
        self.assertTrue(any(s.name == "calibrate_set_size" for s in steps))
        self.assertEqual(steps[-1].name, "validate_submission")
        self.assertTrue(str(calib_path).endswith("calibration/calibration.json"))
        self.assertTrue(str(submission_csv).endswith("submission_ensemble.csv"))

        pseudo_steps = [s for s in steps if s.name.startswith("pseudo_label_fold")]
        self.assertTrue(len(pseudo_steps) > 0)
        for step in pseudo_steps:
            cmd = step.cmd
            self.assertIn("--batch-size", cmd)
            self.assertIn("--num-workers", cmd)
            self.assertIn("256", cmd)
            self.assertIn("2", cmd)


if __name__ == "__main__":
    unittest.main()
