
# Swimmer_v3 using stable-baselines3 



This is a test of a `TD3` agent playing `Swimmer_v3` using `stable-baselines3` library and the `RL Zoo`.




## Github repositories

 - [RL Zoo](https://github.com/DLR-RM/rl-baselines3-zoo)
 - [SB3](https://github.com/DLR-RM/stable-baselines3)
 - [SB3 Contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib)


## Installation

To run this part of the project, you will need to install the libraries mentionned above. 

### Install RL Zoo

#### Minimal installation

```bash
git clone https://github.com/DLR-RM/rl-baselines3-Zoo
cd rl-baselines3-Zoo
sudo pip install -e .
```

#### Full installation (with extra envs and test dependencies)

```bash
sudo apt-get install swig cmake ffmpeg
sudo pip install -r requirements.txt
```

### Install SB3

To install SB3 with pip, execute:

```bash
sudo pip install stable-baselines3[extra]
```

### Install SB3 Contrib

To install SB3 Contrib with pip, execute:

```bash
sudo pip install sb3-contrib
```
Now that all the needed libraries are installed, you can train an agent.




## Training

To train a TD3 agent on the Swimmer_v3 environnement, follow these simple steps:

```bash
rl_zoo3 train --algo td3 --env Swimmer-v3 -f logs/
```

Now that you have a trained model, if you want to visualize it, simply run:

```bash
rl_zoo3 enjoy --algo td3 --env Swimmer-v3  -f logs/
```


## Our results

This is the model we had after training a TD3 agent playing Swimmer-v3, using the optimum hyperparameters for that specific environnement.

### Hyperparameters

Source: https://huggingface.co/sb3/td3-Swimmer-v3

```bash
OrderedDict([('gamma', 0.9999),
             ('gradient_steps', 1),
             ('learning_starts', 10000),
             ('n_timesteps', 1000000.0),
             ('noise_std', 0.1),
             ('noise_type', 'normal'),
             ('policy', 'MlpPolicy'),
             ('train_freq', 1),
             ('normalize', False)])
```

### Curves

#### reward




