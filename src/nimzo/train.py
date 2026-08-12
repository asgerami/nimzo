import os
import sys
import math
import argparse
import yaml
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from pathlib import Path
from nimzo.model import build_model_from_config, ChessResNet
from nimzo.data.shards import ShardDataset, RatioSampler

def _cosine_warmup_lr_lambda(step: int, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, progress)
    return 0.5 * (1.0 + math.cos(math.pi * progress))

def train(config_path: str, data_path: str, run_name: str = "run_base", resume: str = None, overfit_subset: int = None, device: str = None, steps: int = 10000, eval_data_path: str = None, puzzle_ratio: float = 0.4):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    if device.type == "cpu":
        # Enforce physical thread count (6 cores on Ryzen 5)
        physical_cores = 6
        torch.set_num_threads(physical_cores)
        print(f"Set PyTorch CPU threads to physical count: {torch.get_num_threads()}")
    else:
        print(f"Training on device: {torch.cuda.get_device_name(device)}")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    run_dir = Path(f"runs/{run_name}")
    run_dir.mkdir(parents=True, exist_ok=True)

    model = build_model_from_config(config_path)
    model.to(device)
    model.train()

    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("lr", 1e-3)),
        weight_decay=float(cfg.get("weight_decay", 1e-4))
    )

    # Cosine schedule with linear warmup, over the full planned run (T_max =
    # `steps`). Must be created before the resume block below so a resumed
    # run's scheduler.load_state_dict() picks up the same horizon rather than
    # restarting warmup each session.
    max_steps = 100 if overfit_subset else steps
    warmup_steps = max(1, int(0.03 * max_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: _cosine_warmup_lr_lambda(s, warmup_steps, max_steps)
    )

    start_step = 0
    if resume and os.path.exists(resume):
        print(f"Resuming training from checkpoint: {resume}")
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt.get("step", 0)
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        elif start_step > 0:
            # Older checkpoint saved before the LR schedule existed -- fast-
            # forward so the schedule still roughly matches the true step
            # count instead of silently restarting warmup.
            for _ in range(start_step):
                scheduler.step()

    dataset = ShardDataset(data_path)
    if overfit_subset is not None and overfit_subset < len(dataset):
        print(f"Running overfit test on subset of {overfit_subset} positions.")
        indices = list(range(overfit_subset))
        dataset = torch.utils.data.Subset(dataset, indices)
        loader = DataLoader(
            dataset,
            batch_size=cfg.get("batch_size", 512),
            shuffle=True,
            num_workers=2,
            persistent_workers=False
        )
    elif eval_data_path:
        # Ratio-controlled sampling across two data sources. Plain
        # shuffle-over-concatenation (the old behavior when both files were
        # just cat'd together) samples uniformly over raw record count, so
        # the smaller source gets proportionally drowned out as the larger
        # one grows -- this is what caused puzzle policy accuracy to collapse
        # from 59% to 39% when 5M eval positions were added to 900K puzzles.
        eval_dataset = ShardDataset(eval_data_path)
        n_puzzle = len(dataset)
        n_eval = len(eval_dataset)
        combined = ConcatDataset([dataset, eval_dataset])
        sampler = RatioSampler(n_puzzle, n_eval, ratio_a=puzzle_ratio, num_samples=len(combined))
        print(f"Data sources: {n_puzzle:,} puzzle positions, {n_eval:,} eval positions, target puzzle ratio {puzzle_ratio:.0%}")
        dataset = combined
        loader = DataLoader(
            dataset,
            batch_size=cfg.get("batch_size", 512),
            sampler=sampler,
            num_workers=2,
            persistent_workers=True
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=cfg.get("batch_size", 512),
            shuffle=True,
            num_workers=2,
            persistent_workers=True if len(dataset) > 1000 else False
        )
    
    step = start_step

    print("Starting training loop...")
    t0 = time.time()
    positions_processed = 0

    for epoch in range(1, 1000):
        for planes, actions, wdl_targets, sources in loader:
            planes = planes.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)
            wdl_targets = wdl_targets.to(device, non_blocking=True)
            sources = sources.to(device, non_blocking=True)

            optimizer.zero_grad()

            p_logits, v_logits = model(planes)

            # Policy Loss: Cross entropy against action index
            p_loss = F.cross_entropy(p_logits, actions)

            # Value Loss: Cross entropy against 3-class WDL soft targets
            v_loss = F.cross_entropy(v_logits, wdl_targets)

            total_loss = p_loss + 1.0 * v_loss
            total_loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            step += 1
            positions_processed += len(actions)

            if step % 10 == 0 or step == max_steps:
                elapsed = time.time() - t0
                pos_per_sec = positions_processed / elapsed if elapsed > 0 else 0
                correct = p_logits.argmax(dim=-1) == actions
                acc = correct.float().mean().item()
                # source: 0 = puzzle, 1 = eval DB (see encode_record/shards.py)
                puzzle_mask = sources == 0
                eval_mask = sources == 1
                puzzle_acc = correct[puzzle_mask].float().mean().item() if puzzle_mask.any() else float("nan")
                eval_acc = correct[eval_mask].float().mean().item() if eval_mask.any() else float("nan")
                lr = scheduler.get_last_lr()[0]
                print(f"Step {step}/{max_steps} | LR: {lr:.2e} | Loss: {total_loss.item():.4f} (P: {p_loss.item():.4f}, V: {v_loss.item():.4f}) | Policy Acc: {acc*100:.2f}% (puzzle: {puzzle_acc*100:.2f}%, eval: {eval_acc*100:.2f}%) | {pos_per_sec:.1f} pos/sec")

            if step % 1000 == 0:
                ckpt_path = run_dir / f"ckpt_{step}.pt"
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "step": step,
                }, ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}")
                
            if step >= max_steps:
                break
        if step >= max_steps:
            break
            
    print(f"Training completed at step {step}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Nimzo ResNet engine")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to config YAML")
    parser.add_argument("--data", type=str, required=True, help="Path to binary shard file")
    parser.add_argument("--name", type=str, default="run_base", help="Run identifier")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint file path to resume")
    parser.add_argument("--overfit", type=int, default=None, help="Overfit subset size for M3 gate")
    parser.add_argument("--device", type=str, default=None, help="Force device: 'cpu' or 'cuda'. Default: auto-detect.")
    parser.add_argument("--steps", type=int, default=10000, help="Total training steps (ignored when --overfit is set).")
    parser.add_argument("--eval-data", type=str, default=None, help="Optional path to eval-DB shard file. If given, samples are drawn from --data (puzzles) and this file together at --puzzle-ratio instead of a plain concatenation.")
    parser.add_argument("--puzzle-ratio", type=float, default=0.4, help="Target fraction of each batch drawn from --data when --eval-data is also given.")
    args = parser.parse_args()
    train(args.config, args.data, run_name=args.name, resume=args.resume, overfit_subset=args.overfit, device=args.device, steps=args.steps, eval_data_path=args.eval_data, puzzle_ratio=args.puzzle_ratio)
