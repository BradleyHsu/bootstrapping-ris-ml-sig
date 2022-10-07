## Running MNIST on standalone lab GPU (albatross)

This tutorial uses "mnist.py" script for example. 


Note: Before following this, make sure you have home account created for you. Request [Dr. Jacobs](jacobsn@wustl.edu) for this. To login, you should be on WashU WiFi connection or can use VPN. Instructions for VPN connection [here](https://computing.artsci.wustl.edu/connect-network-through-vpn).


### 1) Remote login

```
ssh [user]@albatross.engr.wustl.edu
```

### 2) Install miniconda

* Download and install

```
wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh

bash Miniconda3-py38_4.12.0-Linux-x86_64.sh

source ~/.bashrc

```
After successful installation you should see `base` activated by default.

* Create and activate conda environment for the project

```
cd ./bootstrapping/mnist_standalone

conda env create -f environment.yml

conda activate project_mnist

```
`project_mnist` is the name of the environment as defined in `environment.yml`.

Note: Step 2 is a one-time process. Once miniconda is installed, for each new project that has an `environment.yml` file,  you can create and activate conda environment accordingly.


### 3) Run training script

`python mnist.py`

This script downloads data, trains the model, evaluates it and stores logs in `./lightning_logs/` folder.