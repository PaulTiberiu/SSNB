# RL Project - Scalability

## Goal

This project aims to study the scalability of RL algorithms when the size of the problem is increased. Here, we consider the gym-based environment `Swimmer` that we tweaked to make it have any finite number of bodies.

## How to install bbrl, bbrl_gym, bbrl_examples and mujoco
### Install bbrl

After cloning the Github repository, you should install the required libraries. Here is the whole process :
```
git clone https://github.com/osigaud/bbrl
cd bbrl
pip install -r requirements.txt
sudo pip install -e .
```


### Install bbrl_gym (only needed for the notebooks' scripts)

The steps are similar to those of bbrl. They are the following :
```
git clone https://github.com/osigaud/bbrl_gym
cd bbrl_gym
pip install -r requirements.txt
sudo pip install -e .
```

### Install bbrl_examples
For bbrl_examples, the steps are also similar:
```
git clone https://github.com/osigaud/bbrl_examples
cd bbrl_examples
pip install -r requirements.txt
sudo pip install -e .
```

### Install mujoco

You should first visit `MuJoCo`'s repository : `https://github.com/openai/mujoco-py#install-mujoco` and download `MuJoCo`. Keep in mind that to proceed further in the procedure, your files should be located in the `home` directory.

Decompress the downloaded archive and type the following command : 
```
mkdir ./mujoco
```

In order to check if the directory has been sucessfully created, type :
```
ls -la
```

Then, move the extracted file into the directory you have just created :
```
mv mujoco210 ./mujoco
```

After you are done, you need to add the following lines at the beginning of your `.bahsrc` file :
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/$USERNAME$/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
```

Please note that `$USERNAME$` should be replaced with your own username. Remember that to check the whole path, you can go into `MuJoCo`'s bin file and then use the `pwd` command.

To complete the installation, type :
```
# Make sure your python environment is activated
sudo pip install -U 'mujoco-py<2.2,>=2.1'
```

NB : the following libraries need to be installed first in order for `MuJoCo` to work properly :
```
sudo apt-get install patchelf
sudo apt-get install libosmesa6-dev
```

### Another useful installations:

We also recommend you, in order to be sure that everything works, to make the following installation :
```
pip install hydra-core --upgrade --pre
```

## How to Run our Program
Now, with everything installed, we can use reinforcement learning algorithms to run different environments. Here we will focus on the Swimmer environment.

### XML file for Swimmer

In order to work, the Swimmer environment needs a XML file. We have already prodided you a bunch of XML files located in the `assets` directory.

### Algorithms used in the Swimmer environment

For now, only two algorithms are at your disposal : `TD3` and `DDPG`. Their scripts can be found in the `algos` directory.

### Run the Swimmer environment

In order to run a Swimmer environment, you only need to execute the algorithms' scripts mentionned beforehand. You can change the number of joints and bodies by changing the `config_name` variable located near the end of the scripts. E.G. : in `td3.py` you can replace the default `config_name` by `td3_swimmer6.yaml` in order to get a swimmer with 6 bodies