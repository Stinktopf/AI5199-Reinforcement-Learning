# Cat Mouse Markov Game

This repository contains a simple Markov game where a cat tries to catch a mouse, and the mouse aims to reach two pieces of cheese. The code implements the Q-learning algorithm and includes hyperparameter optimization using Ray Tune.

## Usage

Install dependencies:

```py
pip install -r requirements.txt
```

Adjust the absolute Ray Tune path to your needs:
```py
ray.init(_temp_dir="C:/ray_temp", ignore_reinit_error=True)
```

Execute:

```py
python Cat_Mouse_Markov_Game.py
```

Or open the Jupyter Notebook.