# Overcooked! - Develop diverse cooking tasks for multi-agent cooperation – Variant 3

For KCL BSc Computer Science (Artificial Intelligence) Dissertation

Contents:

- [Abstract](#0.Abstract)
- [Installation](#1.installation)
- [Bayesian Delegation](#2.BayesianDelegation)
- [DQN](#3.DQN)
<!-- - [Environments and Recipes](docs/environments.md)
- [Design and Customization](docs/design.md) -->

## 0.Abstract

The Gifs are directly taken from the original "Too Many Cook!" Github repository, which can be accessed here: https://github.com/rosewang2008/gym-cooking

<p align="center">
    <img src="images/2_open_salad.gif" width=260></img>
    <img src="images/2_partial_tl.gif" width=260></img>
    <img src="images/2_full_salad.gif" width=260></img>
</p>

Multi-Agent Reinforcement Learning (MARL) is a topic long-discussed in machine learning research, which covers various applications ranging from healthcare systems to automated logistics design. With a publication "Too Many Cooks!" \cite{mitOvercooked} carried out by MIT and Harvard students that discusses agent coordination in partially observable situations, this project expands upon the concept of "Role Differentiation", attempting to observe the effect of different algorithms on role allocation and its impact on the original algorithm for subtask achievements.

This project presents the specific implementation and evaluation of the role allocation algorithm on expanded action sets, covering the comparison between the original simulation, role-optimized solution, and other proposed solutions. Subsequently, the project explores the usage of Deep-Q Learning on multiple agents to compare performances and achieve a different way of agent training. To increase difficulty in agent coordination, the project also explored the possibility of resource scarcity and various agent execution algorithms in subtask selections.

## 1.Installation

### 1.1 Create Environment

Please try with pip3 / python3 if the following does not work on the machine.

```
conda create -n OvercookedV3 python==3.7.16
conda activate OvercookedV3
python -m pip install -e.
cd gym_cooking
```

### 1.2 Tensorflow

To download the tensorflow for deep learning:

```
python -m pip install tensorflow
```

## 2.Bayesian Delegation

### 2.1 Overall Structure

Before starting, please ensure terminal is in the `gym_cooking` repository.

The command structure has the following arguments:

```
python main.py --num-agents <number> --level <level name> --model1 <model name> --model2 <model name> --model3 <model name> --model4 <model name> --role <role name> --record
```

In the original design, the agents were able to choose to run the different models, including:

- `bd` to run Bayesian Delegation,
- `up` for Uniform Priors,
- `dc` for Divide & Conquer,
- `fb` for Fixed Beliefs, and
- `greedy` for Greedy.

However, with the implementation of role differentiation, please run the structure with `bd` only to avoid system issues.

The other parameters include `<number>` for the number of agents, `<level>` for specifying the level that is played, and `<role>`, which can be selected from the following:

- `extreme` for single agents taking all the work,
- `optimal` for the optimal role allocation,
- `unbalanced` for slightly unbalanced work segregation,
- `three` for optimal role separation design,
- `none` for the original non-role specified simulation.

### 2.2 Simulation Command

If wanting to replicate the simulated results in the dissertation, please utilized only 2 agents, and with the levels in `very-easy`, `new-open`, `new-partial`. The four recipes are `tomato`, `salad`, `burger`, and `CF (Chicken and Fish)`. With the combinations of the above roles, can generate 60 results in total.

For example, to run an `unbalanced` role allocaiton with `new-open` level and `burger` recipe, use the following:

```
python main.py --num-agents 2 --level new-open_burger --model1 bd --model2 bd --role unbalanced --record
```

The recorded screenshot at each time step is stored in `misc/game/record/{level name}`, reset at each run.

### 2.3 Manual Play

To manually control the agents with the pygame window, utilize the `--play` flag to the command. Pressing the number `1`, `2`, `3`, and `4` depending on the number of agents will change the agent that can be interacted.

Warning: As the `very-easy` level is comparatively smaller to other environment, it did not specify four agent location, please only utilize `--num-agents 2` in command.
For example, if wanting to play the `new-partial` level with the `CF` recipe, enter the following command:

```
python main.py --num-agents 2 --level new-partial_CF --role none --play
```

If would like to alter the level or create a new level, please create a new `txt` file and specify the 2D environment with the symbols in `core.py`, while specifying the recipe name and the agent location.

### 2.4 Results

If the `pkg` file for a particular run is present in `misc/metrics/pickles`, please delete before proceed on the simulation run, so it would be stored.

Navigate to the `gym_cooking/misc/metrics` directory with the following:

```
cd gym_cooking/misc/metrics
```

If wanting to generate the line graphs, run:

```
python make_graphs.py --completion
```

To generate the legend graphs, run:

```
python make_graphs.py --legend --time-step
```

The results will be stored in `gym_cooking/misc/metrics/graph_agents2`.

## 3.DQN
