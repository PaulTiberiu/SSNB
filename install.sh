echo "Do you wish to proceed with the installation of SSNB ? [y/n] "
read answer

if [answer == 'y' || answer == 'Y']
then
    ssnb_directory = `pwd`

    echo "=== Installing bbrl ===\n"
    git clone https://github.com/osigaud/bbrl
    cd bbrl/
    sudo pip install -e .

    cd ../

    echo "=== Installing dependencies for MuJoCo ===\n"
    sudo apt-get install libglew-dev
    sudo apt-get install libosmesa6-dev
    sudo apt-get install patchelf

    echo "=== Installing MuJoCo ===\n"
    curl https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
    tar -xvf mujoco210-linux-x86_64.tar.gz -C ~/
    rm -rf mujoco210-linux-x86_64.tar.gz
    cd ~/
    mkdir .mujoco/
    mv $ssnb_directory/mujoco210/ .mujoco/

    home_directory = `pwd`
    echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$home_directory/.mujoco/mujoco210/bin" > .bashrc
    echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/lib/nvidia" > .bashrc
    echo "export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so" > .bashrc

    echo "=== Installing SSNB ===\n"
    cd ssnb_directory
    sudo pip install -e .

    echo "Installation successful\n"
    
else
    echo "Installation aborted\n"
fi