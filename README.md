# SSNB

![SSNB Banner](https://github.com/PaulTiberiu/SSNB/blob/main/SSNB.png)

`SSNB` stands for "Scalability for Swimmer with N Bodies". This project aims to study the scalability of RL algorithms when the size of the problem is increased. We thus consider the gym-based environment `Swimmer` that we tweaked to make it have any finite number of bodies. SSNB's code being greatly inspired from `bbrl` and `bbrl_examples`, here are the links to these repositories: [bbrl](https://github.com/osigaud/bbrl) and [bbrl_examples](https://github.com/osigaud/bbrl_examples).


## Run SSNB

### Swimmer with N Bodies

In order to work, the `Swimmer` environment needs a XML file. We have already prodided you a bunch of XML files located in the `assets` directory. Each file matches a `Swimmer` with a number of bodies.


### RL algorithms used in SSNB

Right now, only two algorithms are available: `TD3` and `DDPG`. Their scripts can be found in the `algos` directory.


### Run your own simulation

In order to run a `Swimmer` environment, you only need to execute the algorithms' scripts mentionned beforehand: `python TD3.py`. You can change the number of joints and bodies by changing the `config_name` variable located near the end of the scripts. For example, in `td3.py` you can replace the default `config_name` by `"td3_swimmer6.yaml"` in order to get a `Swimmer` environment with 6 bodies.

If you wish to view your environment, you can set the `render_agents` parameter to `True` in the related config file.


## Hyperparameters' tuning with Optuna

`SSNB` also provides a way to get your own hyperparameters with the help of Optuna. In order to execute the optimization script, you need to run `python optimize.py`. Also, you can change the number of bodies, the span of a trial, or the hyperparameters you want to tune in the config file: `optimize_swimmer.yaml`.


### Detailed Functioning

Our optimization script is based on a config file named `optimize_swimmer.yml` that provides the relevant keys needed to launch a study, i.e.: the total number of trials, the number of seeds, the hyperparameters that need to be tuned, the algorithm and the number of bodies.

At the beginning of the study, we generate a certain number of seeds that are later printed in the shell. Plus, here are the steps for each trial:
- Optuna defines a new set of hyperparameters based on the config file and the range of the values
- We then loop over our seeds and for each we do the following:
- We execute the algorithm and interrupt it at the end of the amount of steps precised in the config file
- We report its score to Optuna. In case it is disappointing, the trial is ended. See [Optuna's Median Pruner](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.MedianPruner.html) for further information.
- When the loop is over, the average of these scores is computed and defined as the trial's value.


### Implementing a new algorithm

In order to add a new algorithm, you have to implement it with OOP and make sure it has a class function `create_agent: cls x cfg -> NewAlgorithmAgent` and a method `run: self -> void` that executes the algorithm for a number of steps given in the config file (in our case: budget).


## Install SSNB

### Using the shell script

Installing `SSNB` is actually quite easy as all you need to do is executing our shell script:
```
chmod u+x install.sh
./install.sh
```


### Manual installation

#### Install bbrl

Alternatively, you have to first install `bbrl`. Here is the whole process:
```
git clone https://github.com/osigaud/bbrl
cd bbrl
sudo pip install -e .
```


#### Update Hydra-Core

In order to make sure that everything works, please update `Hydra-Core`:
```
pip install hydra-core --upgrade --pre
```


#### Install MuJoCo

You can install `MuJoCo` by following the steps described here: [Install MuJoCo](https://github.com/openai/mujoco-py#install-mujoco).

**NB:** the following libraries need to be installed first in order for `MuJoCo` to work properly:
```
sudo apt-get install libgl1-mesa-glx
sudo apt-get install libglew-dev
sudo apt-get install libglfw3
sudo apt-get install libosmesa6-dev
sudo apt-get install patchelf
```


#### Install the SSNB package

Finally, type the following command in `SSNB`'s repository:
```
pip install .
```


### Troubleshooting

#### MuJoCo issues

If you have trouble with `MuJoCo`, you may want to refer to the following links:
- [MuJoCo Troubleshooting](https://github.com/openai/mujoco-py#troubleshooting)
- [Permission Error](https://github.com/openai/mujoco-py/issues/351)


#### Python import errors

If you have import errors when running any RL algorithm, it is probably because the repository has not been appended to your `sys.path`. You can either:

- Reinstall any package that `Python` fails to import. In the correct directory, type:
```
sudo pip install -e .
```

- Or add the package path in the program:
```
# If sys has not been imported previously:
import sys
# Define a package_path variable that refers to the path of the package from which you want to import the modules
sys.path.append(package_path)
```
