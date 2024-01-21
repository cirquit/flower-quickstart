# Flower Quickstart

Based on the flower pytorch quickstart example.

Additions:
- wandb integration
- experiment counting
- SplitLRFedAvg strategy that splits the clients in two groups, one with a constant lr and one with an adaptive one

### Example

```bash
> pip install -r requirements.txt
> mkdir .exp-count
> ./run-server <num-rounds>
> ./run-clients <num-clients>
```

### Wandb setup

```bash
> wandb init
> (paste your API key)
```
