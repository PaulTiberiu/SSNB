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
### Other useful installations:
We also recommend you, in order to be sure that everything works, to make the following installation:
```
pip install hydra-core --upgrade --pre
```
## How To
Now, with everything installed, we can use reinforcement learning algorithms to run different environments. Here we will focus on the Swimmer environment.

### XML file for Swimmer
In order to work, the Swimmer environment needs a XML file. You can find it in the gym folder and it is named swimmer.xml. The path is the following:
```
gym/gym/envs/mujoco/assets
```
NB: The XML file works for a 3 bodied Swimmer, but you can find more examples in the assets folder on this github page.

#### 1st Case

In order to use another XML file, you need to follow this path:
```
gym/gym/envs/mujoco
```
In this folder you need to search the swimmer_v3.py file and you need to replace the following line of code (it can be found at the top of the file),by changing the "swimmer.xml" file with a .xml file of your choice:
```
xml_file="swimmer.xml"
```

#### 2nd Case
The other method would be to change the following function, found in the td3.py file:
```
def make_gym_env(env_name):
    return gym.make(env_name, xml_file="swimmer5.xml")
```
By using the following path:
```
bbrl_examples/bbrl_examples/algos/td3
```

### Algorithms used in the Swimmer environment
#### TD3
TD3 is located in the bbrl_examples file, it's name is td3.py. The path is the following:
```
bbrl_examples/bbrl_examples/algos/td3
```
Then, you need to replace the td3.py file with our version, that you can find in the main brach of this github, in the algos folder.

#### (TD3) Yaml file
Before testing Swimmer with the TD3 algorithm, you will need to create a folder in the path we precised, named configs, where you will add a yaml file, which will help Swimmer to 'swim' correctly by changing the hyperparameters. You can find the yaml file in this github page with the following path:
```
algos/assets
```

### Test the Swimmer environment
#### TD3
To test if the Swimmer environment works, you need to follow this path:
```
bbrl_examples/bbrl_examples/algos/td3
```
Then, execute the td3.py file:
```
python3 td3.py
```
