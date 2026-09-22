#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sdm.utils import ensure_dir, write_json


@dataclass
class Step:
    name: str
    cmd: list[str]
    artifacts: list[Path]
    requires: list[Path] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full V3.1 pipeline unattended (long-run mode)")
    parser.add_argument("--data-root", type=str, default=os.environ.get("GLC_DATA_DIR", "data"))
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--output-root", type=str, default="artifacts/long_run")
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--pseudo-batch-size", type=int, default=256)
    parser.add_argument("--pseudo-num-workers", type=int, default=2)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--resume", dest="resume", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    parser.add_argument("--stop-on-error", dest="stop_on_error", action="store_true")
    parser.add_argument("--no-stop-on-error", dest="stop_on_error", action="store_false")
    parser.set_defaults(stop_on_error=True)
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    return parser.parse_args()


def _lock_or_fail(lock_path: Path) -> None:
    if lock_path.exists():
        raise RuntimeError(f"Lock file already exists: {lock_path}. Another long run may be active.")
    lock_payload = {
        "pid": os.getpid(),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with lock_path.open("w", encoding="utf-8") as handle:
        json.dump(lock_payload, handle, indent=2)


def _unlock(lock_path: Path) -> None:
    if lock_path.exists():
        lock_path.unlink()


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _run_cmd(step: Step, env: dict[str, str], log_jsonl: Path) -> int:
    start = time.time()
    print(f"\n[STEP] {step.name}")
    print("$ " + " ".join(step.cmd), flush=True)

    proc = subprocess.Popen(
        step.cmd,
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
    rc = proc.wait()

    elapsed = time.time() - start
    _append_jsonl(
        log_jsonl,
        {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "step": step.name,
            "cmd": step.cmd,
            "return_code": int(rc),
            "elapsed_sec": float(elapsed),
        },
    )
    return rc


def _make_steps(args: argparse.Namespace) -> tuple[list[Step], list[Path], Path, Path]:
    out_root = Path(args.output_root)
    audit_json = out_root / "audit" / "modality_coverage.json"
    train_pa_dir = out_root / "train_pa"
    pseudo_dir = out_root / "pseudo"
    train_student_dir = out_root / "train_student"
    calib_path = out_root / "calibration" / "calibration.json"
    submission_csv = out_root / "submission_ensemble.csv"

    steps: list[Step] = []
    student_ckpts: list[Path] = []

    steps.append(
        Step(
            name="audit_modalities",
            cmd=[
                args.python_bin,
                str(ROOT / "scripts" / "audit_modalities.py"),
                "--data-root",
                args.data_root,
                "--config",
                args.config,
                "--output-json",
                str(audit_json),
            ],
            artifacts=[audit_json],
            requires=[],
        )
    )

    for fold in args.folds:
        teacher_ckpt = train_pa_dir / f"teacher_fold{fold}.pt"
        pseudo_labels = pseudo_dir / f"pseudo_labels_fold{fold}.pt"
        pseudo_stats = pseudo_dir / f"pseudo_labels_fold{fold}_stats.json"
        student_ckpt = train_student_dir / f"student_fold{fold}.pt"

        steps.append(
            Step(
                name=f"train_pa_fold{fold}",
                cmd=[
                    args.python_bin,
                    str(ROOT / "scripts" / "train_pa.py"),
                    "--data-root",
                    args.data_root,
                    "--config",
                    args.config,
                    "--fold",
                    str(fold),
                    "--output-dir",
                    str(train_pa_dir),
                ],
                artifacts=[teacher_ckpt],
                requires=[],
            )
        )
        steps.append(
            Step(
                name=f"pseudo_label_fold{fold}",
                cmd=[
                    args.python_bin,
                    str(ROOT / "scripts" / "pseudo_label_po.py"),
                    "--teacher-checkpoint",
                    str(teacher_ckpt),
                    "--data-root",
                    args.data_root,
                    "--config",
                    args.config,
                    "--output",
                    str(pseudo_labels),
                    "--stats-json",
                    str(pseudo_stats),
                    "--batch-size",
                    str(int(args.pseudo_batch_size)),
                    "--num-workers",
                    str(int(args.pseudo_num_workers)),
                ],
                artifacts=[pseudo_labels, pseudo_stats],
                requires=[teacher_ckpt],
            )
        )
        steps.append(
            Step(
                name=f"train_student_fold{fold}",
                cmd=[
                    args.python_bin,
                    str(ROOT / "scripts" / "train_student.py"),
                    "--teacher-checkpoint",
                    str(teacher_ckpt),
                    "--pseudo-labels",
                    str(pseudo_labels),
                    "--data-root",
                    args.data_root,
                    "--config",
                    args.config,
                    "--output-dir",
                    str(train_student_dir),
                ],
                artifacts=[student_ckpt],
                requires=[teacher_ckpt, pseudo_labels],
            )
        )
        student_ckpts.append(student_ckpt)

    steps.append(
        Step(
            name="calibrate_set_size",
            cmd=[
                args.python_bin,
                str(ROOT / "scripts" / "calibrate_set_size.py"),
                "--checkpoints",
                *[str(x) for x in student_ckpts],
                "--data-root",
                args.data_root,
                "--output-json",
                str(calib_path),
            ],
            artifacts=[calib_path],
            requires=list(student_ckpts),
        )
    )

    steps.append(
        Step(
            name="predict_test_ensemble",
            cmd=[
                args.python_bin,
                str(ROOT / "scripts" / "predict_test_ensemble.py"),
                "--checkpoints",
                *[str(x) for x in student_ckpts],
                "--data-root",
                args.data_root,
                "--calibration-json",
                str(calib_path),
                "--output-csv",
                str(submission_csv),
            ],
            artifacts=[submission_csv],
            requires=[*student_ckpts, calib_path],
        )
    )

    steps.append(
        Step(
            name="validate_submission",
            cmd=[
                args.python_bin,
                str(ROOT / "scripts" / "validate_submission.py"),
                "--submission",
                str(submission_csv),
            ],
            artifacts=[submission_csv],
            requires=[submission_csv],
        )
    )

    return steps, student_ckpts, calib_path, submission_csv


def main() -> None:
    args = parse_args()

    out_root = ensure_dir(args.output_root)
    lock_path = out_root / ".long_run.lock"
    log_jsonl = out_root / "run_log.jsonl"
    state_json = out_root / "run_state.json"

    _lock_or_fail(lock_path)
    try:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        env["PYTHONPATH"] = str(ROOT)
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONNOUSERSITE"] = "1"
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["NUMEXPR_NUM_THREADS"] = "1"

        steps, student_ckpts, calib_path, submission_csv = _make_steps(args)

        state: dict[str, Any] = {
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "resume": bool(args.resume),
            "stop_on_error": bool(args.stop_on_error),
            "max_retries": int(args.max_retries),
            "output_root": str(out_root),
            "overrides": {
                "pseudo_batch_size": int(args.pseudo_batch_size),
                "pseudo_num_workers": int(args.pseudo_num_workers),
            },
            "steps": [],
        }
        write_json(state, state_json)

        for step in steps:
            step_state = {
                "name": step.name,
                "artifacts": [str(p) for p in step.artifacts],
                "status": "pending",
                "attempts": 0,
            }

            if args.resume and step.artifacts and all(p.exists() for p in step.artifacts):
                step_state["status"] = "skipped_existing"
                state["steps"].append(step_state)
                write_json(state, state_json)
                _append_jsonl(
                    log_jsonl,
                    {
                        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "step": step.name,
                        "status": "skipped_existing",
                    },
                )
                print(f"[SKIP] {step.name} (artifacts exist)")
                continue

            missing_inputs = [str(p) for p in step.requires if not p.exists()]
            if missing_inputs:
                step_state["status"] = "failed_missing_inputs"
                step_state["missing_inputs"] = missing_inputs
                state["steps"].append(step_state)
                write_json(state, state_json)
                msg = f"[ERROR] {step.name} missing required inputs: {missing_inputs}"
                print(msg)
                _append_jsonl(
                    log_jsonl,
                    {
                        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "step": step.name,
                        "status": "failed_missing_inputs",
                        "missing_inputs": missing_inputs,
                    },
                )
                if args.stop_on_error:
                    raise RuntimeError(msg)
                continue

            success = False
            for attempt in range(args.max_retries + 1):
                step_state["attempts"] = attempt + 1
                rc = _run_cmd(step=step, env=env, log_jsonl=log_jsonl)
                if rc == 0:
                    success = True
                    break
                print(f"[WARN] Step failed (attempt {attempt + 1}/{args.max_retries + 1}): {step.name}")

            if success:
                step_state["status"] = "done"
                if step.artifacts and not all(p.exists() for p in step.artifacts):
                    step_state["status"] = "failed_missing_artifact"
            else:
                step_state["status"] = "failed"

            state["steps"].append(step_state)
            write_json(state, state_json)

            if step_state["status"].startswith("failed"):
                msg = f"[ERROR] {step.name} failed after retries"
                print(msg)
                _append_jsonl(
                    log_jsonl,
                    {
                        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "step": step.name,
                        "status": step_state["status"],
                    },
                )
                if args.stop_on_error:
                    raise RuntimeError(msg)

        state["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        state["student_checkpoints"] = [str(x) for x in student_ckpts]
        state["calibration_json"] = str(calib_path)
        state["submission_csv"] = str(submission_csv)
        write_json(state, state_json)

        print("\n[DONE] Long-run pipeline completed")
        print(f"State: {state_json}")
        print(f"Logs: {log_jsonl}")
        print(f"Submission: {submission_csv}")
    finally:
        _unlock(lock_path)


if __name__ == "__main__":
    main()
