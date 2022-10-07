## Running MNIST on RIS with a single GPU

This tutorial uses "ris_mnist.py" script for example. 
There are two ways to run this script on RIS. Interactive mode and non Interactive Mode.


Note: Before following this, make sure you copy the script in your Active user directory and add the following line in your bashrc file
```
export LSF_DOCKER_VOLUMES="/storage1/fs1/jacobsn/Active/[user_directory]:/storage1/fs1/jacobsn/Active/[user_directory] /scratch1/fs1/jacobsn:/scratch1/fs1/jacobsn $HOME:$HOME"
```

### 1) Interactive Mode
Run the following commands:
```
export LSF_DOCKER_SHM_SIZE=8g

bsub -Is -q general-interactive -R "select[gpuhost] rusage[mem=32000] span[hosts=1]" -gpu "num=1" -a 'docker(adhakal2/mvrl:pl)' /bin/bash

python3 ris_mnist.py
```

### 2) Non-interactive Mode
Run the following commands

```
export LSF_DOCKER_SHM_SIZE=8g

bsub -q general -J mnist_ris -R "select[gpuhost] rusage[mem=32000] span[hosts=1]" -oo $ACTIVE/[path_to_output_file] -gpu "num=1" -a 'docker(adhakal2/mvrl:pl)' python3 ris_mnist.py
```

### Commonly used LSF flags
-q: specifies the queue
 
-Is: to run in interactive mode

-R: allows you to specipy multiple resources

-gpu: set gpu specific parameters

-a: specify docker container

-m: specify the host to run the job on. If a docker container is already downloaded in this node, it is faster to run jobs in the same node

-J: specify job name

-oo: specify file_path to dump output

### Useful LSF commands
bsub: run job

bjobs: view currently running jobs

bkill [job_id]: kill a job

bqueues: view all queues

bhosts: view all hosts

bhosts -gpu -w [queue]: view all gpu hosts in a queue #for some reason only works with general-interactive


