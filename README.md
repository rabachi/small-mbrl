# small-mbrl
Experiments for Optimistic Risk-Aware Model-based RL.
Bayesian MBRL with various planning strategies. 

To setup the dependencies:

```conda env create -f conda_env.yml```


To train "CVaR-constrained Upper CVaR" on Distributional-Shift environment:

```python main2.py train_type=upper-cvar-opt-cvar hydra.run.dir=output/${now:%Y-%m-%d}```

Logging is done through Weights and Biases on the public project: 'rabachi/small_mbrl'
