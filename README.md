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
qsub .........
```

Find its name with
```shell
qstat -n
```
Assuming the name is `r10n5` we can set up the second tunnel with

```shell
ssh r10n5 -L 8888:localhost:8888
```

Finally,
```shell
cd $VSC_DATA
source activate tensorflow_non_gpu
jupyter notebook --port=8888
```
and copy-paste the link in the first lines of the output to go to your notebook server.

