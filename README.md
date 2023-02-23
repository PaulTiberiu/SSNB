# SSNB

![SSNB Banner](https://github.com/PaulTiberiu/SSNB/blob/main/SSNB.png)

`SSNB` stands for "Scalability for Swimmer with N Bodies". This project aims to study the scalability of RL algorithms when the size of the problem is increased. We thus consider the gym-based environment `Swimmer` that we tweaked to make it have any finite number of bodies. SSNB's code being greatly inspired from `bbrl` and `bbrl_examples`, here are the links to these repositories: [bbrl](https://github.com/osigaud/bbrl) and [bbrl_examples](https://github.com/osigaud/bbrl_examples).


## Run SSNB

### Swimmer with N Bodies

In order to work, the `Swimmer` environment needs a XML file. We have already prodided you a bunch of XML files located in the `assets` directory. Each file matches a `Swimmer` with a number of bodies.


### RL algorithms used in SSNB

Right now, only two algorithms are available: `TD3` and `DDPG`. Their scripts can be found in the `algos` directory.


### Run your own simulation

In order to run a `Swimmer` environment, you only need to execute the algorithms' scripts mentionned beforehand. You can change the number of joints and bodies by changing the `config_name` variable located near the end of the scripts. For example, in `td3.py` you can replace the default `config_name` by `"td3_swimmer6.yaml"` in order to get a `Swimmer` environment with 6 bodies.

If you want to render the agent, you have to set the `render_agents` parameter to `True` in the related config file. Please note that it may slow down the simulation.


## Install SSNB

### Using the bash script

Installing `SSNB` is actually quite easy as all you need to do is executing our shell script:
```
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
sudo apt-get install libglfw3
sudo apt-get install libosmesa6-dev
sudo apt-get install patchelf
```


#### Install the SSNB package

Finally, type the following command in `SSNB`'s repository:
```
sudo pip install -e .
```


### Troubleshooting

#### Script issues

If you can't run the `install.sh` executable, please retry after typing the following command in `SSNB`'s directory:
```
chmod u+x install.sh
```


#### MuJoCo issues

If you have trouble with `MuJoCo`, you may want to refer to the following links:
- [MuJoCo Troubleshooting](https://github.com/openai/mujoco-py#troubleshooting)
- [Permission Error](https://github.com/openai/mujoco-py/issues/351)


#### Python import errors

If you have import errors when running any RL algorithm, it is probably because the repository has not been appended to your `sys.path`. You can:

1. Reinstall any package that `Python` fails to import. In the correct directory, type:
```
sudo pip install -e .
```

2. Add the package path in the program:
```
# If sys has not been imported previously:
import sys
# Define a package_path variable that refers to the path of the package from which you want to import the modules
sys.path.append(package_path)
```
