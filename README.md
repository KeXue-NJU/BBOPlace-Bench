# BBOPlace-Bench: Benchmarking Black-Box Optimization for Chip Placement

Official implementation of paper "BBOPlace-Bench: Benchmarking Black-Box Optimization for Chip Placement".

This repository contains the Python code for BBOPlace-Bench, a benchmarking containing BBO algorithms for chip placement tasks. 

## Requirements
+ pyyaml==6.0.2
+ psutil==5.7.2
+ ray==2.3.0
+ ray[tune]
+ pandas==2.0.3
+ wandb==0.17.0
+ torch==1.7.1
+ torchelastic==0.2.1
+ torchvision==0.8.2
+ torchaudio==0.7.2
+ matplotlib==3.4.3
+ shapely==2.0.6
+ cairocffi==1.2.0
+ networkx==3.1
+ igraph==0.11.6
+ pymoo==0.6.1.1
+ gpytorch==1.4.0
+ botorch==0.3.3
+ pypop7==0.0.82

## File structure

+ `benchmarks` directory stores the benchmarks for running. Please download ISPD2005 and ICCAD2015 benchmarks and move them to `benchmarks/` (i.e., `benchmarks/ispd2005/adaptec1`, `benchmark/iccad2015/superblue1`).
+ `config` stores the hyperparameters for algorithms.
+ `script` contains scripts for code running.
+ `src` contains the source code of our benchmarking.
+ `thirdparty` serves as a thirdparty standard placer borrowed from [DREAMPlace](<https://github.com/limbo018/DREAMPlace>).
  
## Usage
Please first build the environment according to the requirements or download docker image from [Baidu Netdisk](https://pan.baidu.com/s/12l2kVdF_9b9hMMdxcT6-rg?pwd=2ymp) and download benchmarks via google drive: [ISPD2005](https://drive.google.com/drive/folders/1MVIOZp2rihzIFK3C_4RqJs-bUv1TW2YT?usp=sharing), [ICCAD2015](https://drive.google.com/file/d/1JEC17FmL2cM8BEAewENvRyG6aWxH53mX/view?usp=sharing).

### Using Docker
Please first install docker and [docker cuda tookit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
Load the docker image `bboplace-bench.tar` and run a container as bellow:
```shell
# load our docker image
docker load --input bboplace-bench.tar

# find our docker image and rename it
docker image
docker tag <IMAGE ID> <Your Name>/bboplace-bench:latest

# run a docker container
docker run --gpus all -it -v $(pwd):/workspace <Your Name>/bboplace-bench:latest bash
```

### Parameters
Before running an experiment, you can modify the hyper-parameters of different problem formulations and algorithms in `config` directory. For example, modifying `n_population : 50` in `config/algorithm/ea.yaml` to change the population size of EA.
If you want to use wandb to store experimental data, please first enter the wandb API key in `config/default.yaml` and set `use_wandb` to `True`. For offline use, set `wandb_offline` to `True`.
```yaml
# wandb
use_wandb : True
wandb_offline : True
wandb_api : <Your API KEY>
```

### Run an experiment

Navigate to the `script` directory.
```shell
cd script
```