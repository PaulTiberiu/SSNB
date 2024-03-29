
# Swimmer_v3 using stable-baselines3 



This is a test of a `TD3` agent playing `Swimmer_v3` using the `stable-baselines3` library and `RL Zoo`.




## Featured Github repositories

 - [RL Zoo](https://github.com/DLR-RM/rl-baselines3-zoo)
 - [SB3](https://github.com/DLR-RM/stable-baselines3)
 - [SB3 Contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib)


## Installation

To run this part of the project, you will need to install the libraries mentioned above. 

### Install RL Zoo

#### Minimal installation

```
git clone https://github.com/DLR-RM/rl-baselines3-Zoo
cd rl-baselines3-Zoo
sudo pip install -e .
```

#### Full installation (with extra envs and test dependencies)

```
sudo apt-get install swig cmake ffmpeg
sudo pip install -r requirements.txt
```

### Install SB3

To install SB3 with pip, execute:

```
sudo pip install stable-baselines3[extra]
```

### Install SB3 Contrib

To install SB3 Contrib with pip, execute:

```
sudo pip install sb3-contrib
```
Now that all the needed libraries are installed, you can train an agent.




## Training

To train a TD3 agent on the Swimmer_v3 environnement, use the following steps:

```
rl_zoo3 train --algo td3 --env Swimmer-v3 -f logs/
```

Now that you have a trained agent, if you want to enjoy it, simply run:

```
# exp-id 0 corresponds to the last experiment, otherwise, you can specify another ID
rl_zoo3 enjoy --algo algo_name --env env_id -f logs/ --exp-id 0
```
## Experiment tracking

RL Zoo offers the possibility to visualize experiment data such as learning curves via [Weights and Biases](https://wandb.ai/)

The following command:
```
rl_zoo3 train --algo td3 --env Swimmer-v3 --track --wandb-project-name sb3
```
yields a tracked experiment at this [URL](https://wandb.ai/openrlbenchmark/sb3/runs/1b65ldmh).

### First use

When you type the commande given above for the first time on your machine, you will have to create a `wandb` account and follow the rest of the instructions given on the terminal

Then, the experiment tracking will be available on your `wandb` home page

## Record a video of an agent

Commands to record an agent, trained or untrained, for a specific number of steps

### Trained agent

Record the latest saved model for 1000 steps
```
python -m rl_zoo3.record_video --algo td3 --env Swimmer-v3 -n 1000
``` 

### Training the agent

Record 1000 steps for each checkpoint, latest and best saved models
```
python -m rl_zoo3.record_training --algo td3 --env Swimmer-v3 -n 1000 -f logs --deterministic
```
This command generates a `mp4` file. To convert it into a `gif` file, add `it` at the end of the command.

## Our results

This is the model we had after training a `TD3` agent playing `Swimmer-v3`, using the optimum hyperparameters for that specific environnement.

The model can be found in the `logs` folder above

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

### Learning curves

Using the command mentioned earlier, we were able to track the reward data throughout the training of our agent.

#### Reward curve

![reward curve](https://github.com/PaulTiberiu/SSNB/blob/main/sb3/reward_curve.png)

#### Run summary

![Run summary](https://github.com/PaulTiberiu/SSNB/blob/main/sb3/run_summary.png)





