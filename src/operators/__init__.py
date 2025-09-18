import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import mutation
import crossover
import sampling

REGISTRY = {}

REGISTRY["sampling"] = {
    "grid_guide" : {
        "single_random": sampling.GrideGuideSingleRandomSampling,
        'random' : sampling.GrideGuideRandomSampling,
        "spiral" : sampling.GrideGuideSpiralSampling,
    },
    "sp" : {
        "random" : sampling.SPRandomSampling,
    },
    "dmp": {
        "random" : sampling.HyperparameterSampling,
    }
}

REGISTRY["mutation"] = {
    "grid_guide" : {
        "dummy" : mutation.DummyMutation,
        "swap" : mutation.GridGuideSwapMutation,
        "shift" : mutation.GridGuideShiftMutation,
        "random_resetting" : mutation.GridGuideRandomResettingMutation,
        "shuffle" : mutation.GridGuideShuffleMutation,
        "pm": mutation.GridGuidePMMutation,
    },
    "sp" : {
        "dummy" : mutation.DummyMutation,
        "inversion" : mutation.SPInversionMutation,
    },
    "dmp": {
        "random_resetting" : mutation.HyperparameterRandomResettingMutation,
    }
}

REGISTRY["crossover"] = {
    "grid_guide" : {
        "dummy" : crossover.DummyCrossover,
        "uniform" : crossover.GridGuideUniformCrossover,
        "sbx": crossover.GuidGuideSBXCrossover
    },
    "sp" : {
        "dummy" : crossover.DummyCrossover,
        "order" : crossover.SPOrderCrossover,
    },
    "dmp": {
        "uniform" : crossover.HyperparameterUniformCrossover,
    }
}