
I coded up [NEAT in pure python](neat.py) as a learning exercise, visualizations by gpt5:

![](lunar_lander_scratch.mp4)
Run `uv run train_lunar_lander.py --stagnation_generations 50` to reproduce.

![](lunar_lander_dense.mp4)
Run `uv run train_lunar_lander.py --dense_initial_connectivity` to reproduce.

```
=== NEAT XOR Results (100 runs, 100 gens max) ===

Our implementation:
  Population size: 150
  Generations:        44.8 (std: 20.8)
  Networks evaluated: 6721 (std: 3121)
  Hidden nodes:       3.52 (std: 1.55)
  Connections:        12.05 (std: 3.86)
  Worst performance:  98 generations (14700 evaluations)
  Success rate: 93/100 (93%)

Our implementation:
  Population size: 10_000
  Generations:        9.9 (std: 1.9)
  Networks evaluated: 1484 (std: 291)
  Hidden nodes:       3.33 (std: 1.11)
  Connections:        10.25 (std: 2.24)
  Worst performance:  15 generations (2250 evaluations)
  Success rate: 100/100 (100%)

Original NEAT paper:
  Population size: 150
  Generations:        32 (std: not given)
  Networks evaluated: 4,755 (std: 2,553)
  Hidden nodes:       2.35 (std: 1.11)
  Connections:        7.48 (std: not given)
  Worst performance:  90 generations (13,459 evaluations)
  Success rate:       100/100 (100%)
```

I tried to match the [paper](https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf) as closely as possible, but the resulting networks reached around ~50% of the efficiency in terms of nodes & connections, so there might still be some bugs or hparam differences.

