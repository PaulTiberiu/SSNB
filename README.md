# How to install bbrl, bbrl_gym and mujoco to use swimmerV3

## Install bbrl
Clone the github repository by using the following command:
```
git clone https://github.com/osigaud/bbrl
```

Then, if the file requirements.txt is not in the bbrl file, install the requirements in the bbrl file of the git you have just cloned:
```
pip install -r requirements.txt
```

After that, also in the bbrl file, you can finish installing bbrl with this command:
```
sudo pip install -e .
```

## Install bbrl_gym

Clone the github repository by using the following command:
```
git clone https://github.com/osigaud/bbrl_gym
```

Then, if the file requirements.txt is not in the bbrl_gym file, install the requirements in the bbrl_gym file of the git you have just cloned:
```
pip install -r requirements.txt
```

After that, also in the bbrl_gym file, you can finish installing bbrl with this command:
```
sudo pip install -e .
```

## Install mujoco

Enter the following website, then download mujoco:
```
https://github.com/openai/mujoco-py#install-mujoco
```
Then, decompress what you have just installed with a right click on the mujoco tar, or use this command:
```
tar -zxvf mujoco210-linux-x86_64.tar.gz
```
After that, you need to create a file with the following command:
```
mkdir ./mujoco
```
To check if it has been created, use the command:
```
ls -la
```
The following step is to move the extracted file into the file you have just created:
```
mv mujoco210 ./mujoco
```
After doing this, you need to change the .bahsrc file by following this steps (you need to be situated in home):
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/tibi/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
```
The first line needs to be replaced with your own path. To check the path, go into the bin file of mujoco and then write in your terminal:
```
pwd
```
