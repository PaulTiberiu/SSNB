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
pip install -e .
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
pip install -e .
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
After that, you need to create an invizible file with the following command:
```
mkdir ./mujoco
```
To check if it has been created, use the command:
```
ls -la
```
The following step is to move the extracted file into the invizible file:
```
mv mujoco210 ./mujoco
```
