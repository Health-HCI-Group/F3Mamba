#!/usr/bin/env python3
import argparse
import os
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser(description="Sanity check for Weights & Biases (wandb).")
    parser.add_argument("--mode", choices=["offline", "online"], default="offline")
    parser.add_argument("--project", default="F3Mamba")
    parser.add_argument("--entity", default=None, help="W&B entity (username or team). Optional.")
    parser.add_argument("--name", default=None, help="Run name. Optional.")
    args = parser.parse_args()

    if args.mode == "offline":
        os.environ.setdefault("WANDB_MODE", "offline")

    if args.mode == "online":
        api_key = os.environ.get("WANDB_API_KEY")
        if not api_key:
            print(
                "WANDB_API_KEY not set. Please run `wandb login` or export WANDB_API_KEY first.",
                file=sys.stderr,
            )
            return 2

    try:
        import wandb
    except Exception as e:
        print(f"Failed to import wandb: {e}", file=sys.stderr)
        return 1

    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=args.name,
        config={
            "sanity": True,
            "mode": args.mode,
            "ts": int(time.time()),
        },
    )

    # Minimal logging
    for step in range(10):
        wandb.log({"step": step, "loss": 1.0 / (step + 1), "lr": 1e-3}, step=step)

    # A small artifact-like file log (optional)
    with open("wandb_sanity_output.txt", "w", encoding="utf-8") as f:
        f.write(f"run_id={run.id}\n")
        f.write(f"mode={args.mode}\n")
        f.write(f"project={args.project}\n")

    wandb.save("wandb_sanity_output.txt")

    run.finish()
    print("wandb sanity check finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
