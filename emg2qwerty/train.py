# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import pprint
from collections.abc import Sequence
from pathlib import Path
from typing import Any
import itertools

import hydra
import pytorch_lightning as pl
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf

from emg2qwerty import transforms, utils
from emg2qwerty.transforms import Transform


log = logging.getLogger(__name__)


GRID = {
    "module.conv_channels":  [[64, 64, 64], [128, 128, 128], [256, 256, 256]],
    "module.rnn_hidden":     [128, 256, 512],
    "module.rnn_layers":     [2, 3],
    "module.dropout":        [0.1, 0.3],
    "optimizer.lr":          [1e-3, 3e-4],
}

def _expand_grid(grid: dict) -> list[dict]:
    """Return a list of dicts, one per combination in the Cartesian product."""
    keys = list(grid.keys())
    values = list(grid.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _apply_overrides(config: DictConfig, overrides: dict) -> DictConfig:
    """Return a *copy* of config with dot-key overrides applied."""
    cfg = config.copy()
    for dotkey, value in overrides.items():
        OmegaConf.update(cfg, dotkey, value, merge=True)
    return cfg

@hydra.main(version_base=None, config_path="../config", config_name="base")
def main(config: DictConfig):
    log.info(f"\nConfig:\n{OmegaConf.to_yaml(config)}")

    # Add working dir to PYTHONPATH
    working_dir = get_original_cwd()
    python_paths = os.environ.get("PYTHONPATH", "").split(os.pathsep)
    if working_dir not in python_paths:
        python_paths.append(working_dir)
        os.environ["PYTHONPATH"] = os.pathsep.join(python_paths)

    # Seed for determinism. This seeds torch, numpy and python random modules
    # taking global rank into account (for multi-process distributed setting).
    # Additionally, this auto-adds a worker_init_fn to train_dataloader that
    # initializes the seed taking worker_id into account per dataloading worker
    # (see `pl_worker_init_fn()`).
    # pl.seed_everything(config.seed, workers=True)

    # Helper to instantiate full paths for dataset sessions
    def _full_session_paths(dataset: ListConfig) -> list[Path]:
        sessions = [session["session"] for session in dataset]
        return [
            Path(config.dataset.root).joinpath(f"{session}.hdf5")
            for session in sessions
        ]

    # Helper to instantiate transforms
    def _build_transform(configs: Sequence[DictConfig]) -> Transform[Any, Any]:
        return transforms.Compose([instantiate(cfg) for cfg in configs])

    # Pre-build data-module once (shared across all grid runs)
    log.info(f"Instantiating LightningDataModule {config.datamodule}")
    datamodule = instantiate(
        config.datamodule,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        train_sessions=_full_session_paths(config.dataset.train),
        val_sessions=_full_session_paths(config.dataset.val),
        test_sessions=_full_session_paths(config.dataset.test),
        train_transform=_build_transform(config.transforms.train),
        val_transform=_build_transform(config.transforms.val),
        test_transform=_build_transform(config.transforms.test),
        _convert_="object",
    )

    callback_configs = config.get("callbacks", [])

    combinations = _expand_grid(GRID)
    log.info(f"Grid search: {len(combinations)} combinations")

    all_results = []
    best_val_cer = float("inf")
    best_combo = None
    best_combo_idx = -1

    for idx, combo in enumerate(combinations):
        log.info(
            f"\n{'='*60}\n"
            f"Grid run {idx + 1}/{len(combinations)}\n"
            f"Hyperparameters: {combo}\n"
            f"{'='*60}"
        )

        run_config = _apply_overrides(config, combo)

        pl.seed_everything(config.seed + idx, workers=True)

        module = instantiate(
            run_config.module,
            optimizer=run_config.optimizer,
            lr_scheduler=run_config.lr_scheduler,
            decoder=run_config.decoder,
            _recursive_=False,
        )

        callbacks = [instantiate(cfg) for cfg in callback_configs]

        run_dir = Path.cwd() / "grid_search" / f"run_{idx:04d}"
        run_dir.mkdir(parents=True, exist_ok=True)

        trainer = pl.Trainer(
            **config.trainer,
            callbacks=callbacks,
        )

        if config.train:
            resume_ckpt = utils.get_last_checkpoint(run_dir / "checkpoints")
            if resume_ckpt:
                log.info(f"Resuming run {idx} from {resume_ckpt}")
            trainer.fit(module, datamodule, ckpt_path=resume_ckpt)
            module = module.load_from_checkpoint(
                trainer.checkpoint_callback.best_model_path
            )

        val_metrics  = trainer.validate(module, datamodule)
        test_metrics = trainer.test(module, datamodule)

        run_result = {
            "run_idx":    idx,
            "hyperparams": combo,
            "val_metrics":  val_metrics,
            "test_metrics": test_metrics,
            "best_checkpoint": trainer.checkpoint_callback.best_model_path,
        }
        all_results.append(run_result)

        val_cer = _extract_cer(val_metrics)
        if val_cer is not None and val_cer < best_val_cer:
            best_val_cer = val_cer
            best_combo = combo
            best_combo_idx = idx

        log.info(f"Run {idx} val CER: {val_cer}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    summary = {
        "best_run_idx":   best_combo_idx,
        "best_val_cer":   best_val_cer,
        "best_hyperparams": best_combo,
        "all_results":    all_results,
    }
    log.info("\n\n=== Grid Search Complete ===")
    pprint.pprint(summary, sort_dicts=False)

    results_path = Path.cwd() / "grid_search_results.yaml"
    OmegaConf.save(OmegaConf.create(summary), results_path)
    log.info(f"Results saved to {results_path}")


def _extract_cer(val_metrics: list[dict]) -> float | None:
    """Pull the first CER-like scalar out of the val_metrics list."""
    if not val_metrics:
        return None
    flat = val_metrics[0] if isinstance(val_metrics, list) else val_metrics
    for key, value in flat.items():
        if "cer" in key.lower():
            return float(value)
    return None

if __name__ == "__main__":
    OmegaConf.register_new_resolver("cpus_per_task", utils.cpus_per_task)
    main()
