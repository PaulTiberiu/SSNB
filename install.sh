#!/bin/bash

while true; do
    read -p "Do you wish to proceed with the installation of SSNB ? [y/n] " answer
    case $answer in
        [Yy]* ) make install; break;;
        [Nn]* ) echo "Installation aborted"; exit;;
        * ) echo "[y/n] ";;
    esac
done

echo "=== Installing setup tools ==="
sudo apt-get install curl
sudo apt-get install git
sudo apt-get install python3-pip

ssnb_directory=$(pwd)

echo "=== Installing bbrl ==="
cd ../
git clone https://github.com/osigaud/bbrl
cd bbrl/
sudo pip install -e .
sudo pip install hydra-core --upgrade --pre

cd ../

echo "=== Installing dependencies for MuJoCo ==="
sudo apt-get install libgl1-mesa-glx
sudo apt-get install libglfw3
sudo apt-get install libosmesa6-dev
sudo apt-get install patchelf
sudo ln -s /usr/lib/x86_64-linux-gnu/libGL.so.1 /usr/lib/x86_64-linux-gnu/libGL.so

echo "=== Installing MuJoCo ==="
curl -o mujoco210-linux-x86_64.tar.gz https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
tar -xvf mujoco210-linux-x86_64.tar.gz -C ~/
rm -rf mujoco210-linux-x86_64.tar.gz
cd ~/
rm -rf .mujoco/
mkdir .mujoco/
mv mujoco210/ .mujoco/

home_directory=$(pwd)
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$home_directory/.mujoco/mujoco210/bin" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/lib/nvidia" >> ~/.bashrc
echo "export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so" >> ~/.bashrc

echo "=== Installing SSNB ==="
cd $ssnb_directory
sudo pip install -e .

echo "Installation successful"
