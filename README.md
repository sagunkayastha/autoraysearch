# autoraysearch

A Claude Code skill for autonomously tuning PyTorch models on a Ray cluster.

Built on top of [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) and [uditgoenka/autoresearch](https://github.com/uditgoenka/autoresearch) — this extends them with Ray cluster support for running the iterate-and-improve loop on GPU hardware.

## How it works

The loop is simple: modify `model.py` → submit to Ray cluster → collect metric → keep or roll back → repeat. Claude handles the ideation and rollback; Ray handles the compute.

## Templates

Two `train.py` templates in `references/train-template.md`:

- **Single-GPU** — wrap an existing `train_main()` function; works with custom DataLoaders, multi-input models, any metric
- **Multi-GPU DDP** — 4-GPU DDP boilerplate for image classification (CIFAR10/100)

## Files

```
SKILL.md                          ← Claude Code skill definition
references/
  train-template.md               ← train.py templates
  ray-cluster.md                  ← Ray cluster setup + known issues (DDP deadlock, cross-node failures)
  setup-protocol.md               ← setup steps
  plan-workflow.md                ← /autoraysearch:plan wizard protocol
  parallel-loop-protocol.md       ← parallel experiment mode
```
