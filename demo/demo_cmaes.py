import os, sys
base_path = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "..")
sys.path.append(base_path)

from config.benchmark import (
    ROOT_DIR, BENCHMARK_DIR
)
THIRDPARTY_DIR = os.path.join(ROOT_DIR, "thirdparty")
SOURCE_DIR = os.path.join(ROOT_DIR, "src")
sys.path.append(ROOT_DIR)
sys.path.append(THIRDPARTY_DIR)
sys.path.append(SOURCE_DIR)
sys.path.append(BENCHMARK_DIR)
os.environ["PYTHONPATH"] = ":".join(sys.path)

import cma 
import random 
import numpy as np 

from src.evaluator import Evaluator

import argparse 
parser = argparse.ArgumentParser() 
parser.add_argument("--sigma", type=float, default=0.5)
parser.add_argument("--pop_size", type=int, default=20)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--placer", type=str, choices=["sp", "grid_guide", "dmp"], default="grid_guide")
parser.add_argument("--benchmark", type=str, default="adaptec1")
parser.add_argument("--eval_gp_hpwl", action="store_true", default=False)
args = parser.parse_args() 

# Set seed
np.random.seed(args.seed)
random.seed(args.seed)

evaluator = Evaluator(args)
dim = evaluator.n_dim
xl = evaluator.xl.tolist() 
xu = evaluator.xu.tolist()
assert len(xl) == len(xu) == dim

x0 = np.random.uniform(low=xl, high=xu, size=dim)

# Initialize CMA-ES
cmaes = cma.CMAEvolutionStrategy(
    x0,  
    args.sigma,  
    {'popsize': args.pop_size,
     'bounds': [xl, xu]}
)

# Run optimization
while not cmaes.stop():
    solutions = cmaes.ask()
    fitness_values = evaluator.evaluate(solutions)
    cmaes.tell(solutions, fitness_values)
    
    with open("results/demo_cmaes.txt", "a") as f:
        f.write(f"Generation {cmaes.countiter}: {cmaes.result.fbest} \n")
    
    if cmaes.countiter % 10 == 0:
        print(f"Generation {cmaes.countiter}: Best fitness = {min(fitness_values):.6f}")

# Print results
print("\nOptimization finished")
print(f"Best solution found: {cmaes.result.xbest}")
print(f"Best fitness: {cmaes.result.fbest}")
print(f"Number of evaluations: {cmaes.result.evaluations}")