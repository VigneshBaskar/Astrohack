# Astrohack

## Team **Brussels Commuters**

## Set up jupyter notebook on a supercomputer over a double tunnel.

First, pick a 4-digit number unique to you (8888 in this example, but pick something else)

### From your local machine:  

```shell
ssh -i id_vcs vsc31950@login.hpc.kuleuven.be -L 8888:localhost:8888
``` 
(or may god help you with setting up port forwarding with putty)

### On the login node

Set up a `conda` environment containing jupyter (for example using one of the environments from [the astrohack repo](https://github.com/gjbex/Astrohack/tree/master/Environments) but adding jupyter in the `.yml`file, or `conda install jupyter` in the environment)

Request your compute note if needed.
```shell
module load tmux
tmux

# For CPU:
qsub -I -A lp_astrohack -lnodes=1:ppn=20 -lwalltime=4:00:00
# For GPU:
qsub -I -A lp_astrohack -lpartition=gpu -lnodes=1:ppn=20:K40c -lwalltime=4:00:00

<CTRL+SPACE> d
```

Find its name with
```shell
qstat -n
```
Assuming the name is `r1i0n1` we can set up the second tunnel with

```shell
module load tmux
tmux
ssh r1i0n1 -L 7333:localhost:7333
# Now you're on the compute node!
cd $VSC_DATA
source activate keras
jupyter notebook --port=7333
<CTRL+SPACE> d
```
and copy-paste the link in the first lines of the output to go to your notebook server.

