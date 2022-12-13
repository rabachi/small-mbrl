# small-mbrl
Experiments for Optimistic Risk-Aware Model-based RL.
Bayesian MBRL with various planning strategies. 

The dependencies can be installed using [conda](https://conda.io/projects/conda/en/latest/index.html):

```conda env create -f conda_env.yml```

```conda activate small_mbrl```


To train "CVaR-constrained Upper CVaR" on Distributional-Shift environment:

```python main2.py train_type=upper-cvar-opt-cvar hydra.run.dir=output/${now:%Y-%m-%d}```

Logging is done through Weights and Biases on the public project: 'rabachi/small_mbrl'
