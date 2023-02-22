# SSNB

![SSNB Banner](https://github.com/PaulTiberiu/SSNB/blob/main/SSNB.png)

`SSNB` stands for "Scalability for Swimmer with N Bodies". This project aims to study the scalability of RL algorithms when the size of the problem is increased. Here, we consider the gym-based environment `Swimmer` that we tweaked to make it have any finite number of bodies.


## Sources

SSNB's code being hugely inspired from `bbrl` and `bbrl_examples`, here are the links to these repositories: [bbrl](https://github.com/osigaud/bbrl) and [bbrl_examples](https://github.com/osigaud/bbrl_examples).


## How to install SSNB

After cloning our Github repository, you first need to type the following command in SSNB's repository:
```
sudo pip install -e .
```


### Install bbrl

Then, you should install the required libraries. Here is the whole process:
```
git clone https://github.com/osigaud/bbrl
cd bbrl
sudo pip install -e .
```


### Install MuJoCo

You can install `MuJoCo` by following the steps described here: [Install MuJoCo](https://github.com/openai/mujoco-py#install-mujoco).

To complete the installation, type:
```
sudo pip install -U 'mujoco-py<2.2,>=2.1'
```

**NB:** the following libraries need to be installed first in order for `MuJoCo` to work properly:
```
sudo apt-get install patchelf
sudo apt-get install libosmesa6-dev
```


### Update Hydra-Core

Also, in order to be sure that everything works, make sure to update `Hydra-Core`:
```
pip install hydra-core --upgrade --pre
```


## How to Run our Program

Now that everything is installed, we can use reinforcement learning algorithms to run different environments. Here we will focus on the `Swimmer` environment.


### XML file for Swimmer

In order to work, the `Swimmer` environment needs a XML file. We have already prodided you a bunch of XML files located in the `assets` directory.


### Algorithms used in the Swimmer environment

For now, only two algorithms are at your disposal: `TD3` and `DDPG`. Their scripts can be found in the `algos` directory.


### Run the Swimmer environment

In order to run a `Swimmer` environment, you only need to execute the algorithms' scripts mentionned beforehand. You can change the number of joints and bodies by changing the `config_name` variable located near the end of the scripts.<br>
For example, in `td3.py` you can replace the default `config_name` by `"td3_swimmer6.yaml"` in order to get a `Swimmer` environment with 6 bodies
