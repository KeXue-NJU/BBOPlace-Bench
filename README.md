# BBOPlace-Bench: Benchmarking Black-Box Optimization for Chip Placement

Official implementation of paper "BBOPlace-Bench: Benchmarking Black-Box Optimization for Chip Placement".

This repository contains the Python code for BBOPlace-Bench, a benchmarking containing BBO algorithms for chip placement tasks. 

## Installation
> **Important Reminder:**
>
> If you find it difficult to install DREAMPlace, we have a basic version available that works out of the box without DREAMPlace. Check it out at: [BBOPlace-miniBench](https://github.com/lamda-bbo/BBOPlace-miniBench)

### Compile DREAMPlace
Please first install docker and [docker cuda toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). Then, please download the docker image from [Baidu Netdisk](https://pan.baidu.com/s/1_0-ZDNUAKdqwyIQWXWdoUQ?pwd=3bef) or [DREAMPlace](<https://github.com/limbo018/DREAMPlace>), and compile `thirdparty/DREAMPlace_source` in the docker container following the below commands:
```
cd thirdparty/DREAMPlace_source
mkdir build
cd build
cmake .. 
make
make install
cd ../..
cp -r DREAMPlace_source/install/dreamplace dreamplace
```

After that, please install the required packages according to `requirements.txt`. Docker container is not required for code running.


## File Structure

+ `benchmarks/` - Stores the benchmarks for running. Please download ISPD2005 and ICCAD2015 benchmarks and move them here.
+ `config/` - Stores the hyperparameters for algorithms.
+ `script/` - Contains scripts for code running.
+ `src/` - Contains the source code of our benchmarking.
+ `thirdparty/` - Serves as a thirdparty standard placer borrowed from [DREAMPlace](https://github.com/limbo018/DREAMPlace).
+ `demo/` - Contains example usage demos.

## Dataset Preparation

Download the required benchmark datasets:
- [ISPD2005 Benchmarks](https://drive.google.com/drive/folders/1MVIOZp2rihzIFK3C_4RqJs-bUv1TW2YT?usp=sharing)
- [ICCAD2015 Benchmarks](https://drive.google.com/file/d/1JEC17FmL2cM8BEAewENvRyG6aWxH53mX/view?usp=sharing)

Extract and place them in the `benchmarks/` directory:
```
benchmarks/
├── ispd2005/
│   ├── adaptec1/
│   ├── adaptec2/
│   └── ...
└── iccad2015/
    ├── superblue1/
    ├── superblue3/
    └── ...
```

## Usage

### Quick Start Demo
We provide a simple demo of how to benchmark BBO algorithms (e.g., CMA-ES) in `demo/demo_cmaes.py`:

```python
from src.evaluator import Evaluator
import numpy as np

# Initialize evaluator
args = SimpleNamespace(
    placer="grid_guide",  # Options: grid_guide, sp, dmp
    benchmark="adaptec1", 
    eval_gp_hpwl=False
)
evaluator = Evaluator(args)

# Get problem dimensions and bounds
dim = evaluator.n_dim
xl, xu = evaluator.xl, evaluator.xu

# Evaluate solutions
x = np.random.uniform(low=xl, high=xu, size=(128, dim))
hpwl, overlap_rate, macro_pos_lst = evaluator.evaluate(x)

# Plot placement figure
evaluator.plot(fig_name="placement.png", macro_pos=macro_pos_lst[0], hpwl=hpwl[0])
```

### Available Benchmarks
```python
benchmarks = [
    # ISPD 2005
    "adaptec1", "adaptec2", "adaptec3", "adaptec4",
    "bigblue1", "bigblue3",
    # ICCAD 2015   
    "superblue1", "superblue3", "superblue4", "superblue5",
    "superblue7", "superblue10", "superblue16", "superblue18"
]
```

### Problem Formulations
- **GG (Grid Guide)**: Continuous optimization
- **SP (Sequence Pair)**: Permutation-based optimization  
- **HPO (DMP)**: Continuous optimization

## Configuration

Modify algorithm hyperparameters in `config/` directory:

```yaml
# config/algorithm/ea.yaml example
n_population: 50
n_generation: 100
```

### Wandb Integration
To use Weights & Biases logging:

```yaml
# config/default.yaml
use_wandb: True
wandb_offline: True  # Set True for offline mode
wandb_api: "<Your API KEY>"
```

**Note**: Set `n_cpu_max=1` when using `dmp` placer or evaluating global placement HPWL.

## Running Experiments

Execute the shell scripts in `script/` directory:
```bash
cd script
./run_experiment.sh
```

<!-- ## Citation
If you find this code useful, please cite our paper:
```bibtex
@article{bboplace2024,
  title={BBOPlace-Bench: Benchmarking Black-Box Optimization for Chip Placement},
  author={...},
  journal={...},
  year={2024}
}
``` -->

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
