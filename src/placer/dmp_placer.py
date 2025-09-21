from .basic_placer import BasicPlacer
from utils.constant import EPS
from thirdparty.dreamplace.Params import Params as DMPParams
from thirdparty.dreamplace.PlaceDB import PlaceDB as DMPPlaceDB
from thirdparty.dreamplace.NonLinearPlace import NonLinearPlace
from typing import Dict, Union
import torch as th
import numpy as np
import math
import os
import signal
import time

from utils.signal_handler import abort_signal_handler, timeout_handler, AbortSignalException
from utils.debug import *

Numeric = Union[int, float]

orig_func = lambda x: x
round_func = lambda x: round(x)
sel_func = lambda options: lambda x: options[math.trunc(x) % len(options)]

params_space = {
###################################################################################
# format:
#   param_tag: (lower_bound, upper_bound, transform_func),
# example:
#   "density_weight": (1e-06, 1.0, orig_func)
###################################################################################

# categorical
    # "GP_num_bins_x": (0, 2, sel_func([1024, 2048])),
    # "GP_num_bins_y": (0, 2, sel_func([1024, 2048])),
    # "GP_optimizer": (0, 2, sel_func(["adam", "nesterov"])),
    # "GP_wirelength": (0, 2, sel_func(["weighted_average", "logsumexp"])),
    # "GP_iteration": (0, 1, sel_func([1000])),
    "GP_num_bins_x": (0, 4, sel_func([256, 512, 1024, 2048])),
    "GP_num_bins_y": (0, 4, sel_func([256, 512, 1024, 2048])),
    "GP_optimizer": (0, 2, sel_func(["adam", "nesterov"])),
    "GP_wirelength": (0, 2, sel_func(["weighted_average", "logsumexp"])),
    "GP_iteration": (0, 1, sel_func([9999])),

# uniform
    # "GP_Llambda_density_weight_iteration": (1, 3, round_func),
    # "GP_Lsub_iteration": (1, 3, round_func),
    # "GP_learning_rate": (0.001, 0.01, orig_func),
    # "GP_learning_rate_decay": (0.99, 1.0, orig_func),
    # "RePlAce_LOWER_PCOF": (0.9, 0.99, orig_func),
    # "RePlAce_UPPER_PCOF": (1.02, 1.15, orig_func),
    # "RePlAce_ref_hpwl": (150000, 550000, round_func),
    # "density_weight": (1e-06, 1e-04, orig_func),
    # "gamma": (1, 4, orig_func),
    # "stop_overflow": (0.06, 0.1, orig_func),
    # "target_density": (0.8, 1.2, orig_func),
    "GP_Llambda_density_weight_iteration": (1, 3, round_func),
    "GP_Lsub_iteration": (1, 3, round_func),
    "GP_learning_rate": (0.001, 0.01, orig_func),
    "GP_learning_rate_decay": (0.99, 1.0, orig_func),
    "RePlAce_LOWER_PCOF": (0.9, 0.99, orig_func),
    "RePlAce_UPPER_PCOF": (1.02, 1.15, orig_func),
    "RePlAce_ref_hpwl": (150000, 550000, round_func),
    "density_weight": (1e-06, 1.0, orig_func),
    "gamma": (0.1, 4, orig_func),
    "stop_overflow": (0.06, 0.1, orig_func),
    "target_density": (0.5, 1.0, orig_func),

}

class DMPPlacer(BasicPlacer):
    DMP_CONFIG_PATH = "config/algorithm/dmp_config"
    DMP_TEMP_BENCHMARK_PATH = "benchmarks/.tmp"
    DMP_RESULT_DIR = os.path.join(
        "results",
        "%(name)s",
        "%(benchmark)s",
        "%(unique_token)s",
        "dmp_results"
    )
    AUX_FILES = [
        "%(benchmark)s.aux",
        "%(benchmark)s.scl",
        "%(benchmark)s.wts",
        "%(benchmark)s.nets",
        "%(benchmark)s.nodes"
    ]
    DEF_FILES = [
        "%(benchmark)s.lef",
        "%(benchmark)s.v",
        "%(benchmark)s.sdc",
        "%(benchmark)s_Early.lib",
        "%(benchmark)s_Late.lib"
    ]

    def __init__(self, args, placedb):
        super(DMPPlacer, self).__init__(args, placedb)
        self.args = args
        self.placedb = placedb
        self.params = DMPParams()
        self.dmp_pldb = DMPPlaceDB()

        # load default dreamplace config
        self._load_dmp_config()

        # prepare benchmark
        self._prepare_benchmark()

        self.params.fromJson(
            {
                "plot_flag": 0,
                "timing_opt_flag": 0,
                "random_seed": self.args.seed,
                "result_dir": self._result_dir,
                "random_center_init_flag": 1,
            }
        )

        self.dmp_pldb(self.params)

        self.dmp_plcr = NonLinearPlace(self.params, self.dmp_pldb, timer=None)


    @property
    def param_dims(self):
        return len(params_space.items())

    @property
    def _result_dir(self):
        ROOT_DIR = self.args.ROOT_DIR
        return os.path.join(
            ROOT_DIR,
            DMPPlacer.DMP_RESULT_DIR % self.args.__dict__
        )

    @property
    def _orig_benchmark_path(self):
        ROOT_DIR = self.args.ROOT_DIR
        return os.path.join(
            ROOT_DIR,
            self.args.benchmark_path
        )

    @property
    def _temp_benchmark_path(self):
        ROOT_DIR = self.args.ROOT_DIR
        return os.path.join(
            ROOT_DIR,
            DMPPlacer.DMP_TEMP_BENCHMARK_PATH,
            "%(benchmark)s_%(unique_token)s" % self.args.__dict__
        )

    def _link_files(self, files):
        for file_name in files:
            orig = os.path.join(
                self._orig_benchmark_path,
                file_name % self.args.__dict__)

            if not os.path.exists(orig):
                continue

            link = os.path.join(
                self._temp_benchmark_path,
                file_name % self.args.__dict__)
            
            os.system(f"ln -sfr {orig} {link}")


    def _prepare_benchmark_aux(self):
        os.makedirs(self._temp_benchmark_path, exist_ok=True)
        self._link_files(DMPPlacer.AUX_FILES)
        
        # prepare .pl
        pl_file_path = os.path.join(
            self._temp_benchmark_path,
            "%(benchmark)s.pl" % self.args.__dict__
        )
        with open(pl_file_path, "w") as pl_file:
            pl_file.write(self.placedb.to_pl(fix_macro=False))

        suffix2path = \
            lambda suffix: os.path.join(
                self._temp_benchmark_path,
                "%(benchmark)s" % self.args.__dict__
            ) + suffix
        self.params.fromJson(
            {
                "aux_input": suffix2path(".aux")
            }
        )
        

    def _prepare_benchmark_def(self):
        os.makedirs(self._temp_benchmark_path, exist_ok=True)
        self._link_files(DMPPlacer.DEF_FILES)
        
        # prepare .def
        def_file_path = os.path.join(
            self._temp_benchmark_path,
            "%(benchmark)s.def" % self.args.__dict__
        )
        with open(def_file_path, "w") as def_file:
            def_file.write(self.placedb.to_def(fix_macro=False))

        suffix2path = \
            lambda suffix: os.path.join(
                self._temp_benchmark_path,
                "%(benchmark)s" % self.args.__dict__
            ) + suffix
        self.params.fromJson(
            {
                "def_input": suffix2path(".def"),
                "lef_input": suffix2path(".lef"),
                "verilog_input": suffix2path(".v"),
                "early_lib_input": suffix2path("_Early.lib"),
                "late_lib_input": suffix2path("_Late.lib"),
                "sdc_input": suffix2path(".sdc")
            }
        )


    def _prepare_benchmark(self):
        type_mapping = {
            "aux": self._prepare_benchmark_aux,
            "def": self._prepare_benchmark_def,
        }
        if self.args.benchmark_type in type_mapping:
            type_mapping[self.args.benchmark_type]()
        else:
            raise NotImplementedError


    def _load_dmp_config(self):
        ROOT_DIR = self.args.ROOT_DIR
        json_file = f"{self.args.benchmark}.json"
        json_path = os.path.join(
            ROOT_DIR,
            DMPPlacer.DMP_CONFIG_PATH,
            json_file
        )
        self.params.load(json_path)
        

    def _load_genotype(self, x):
        params_to_update = {}
        for param_name, value in x.items():
            print(param_name, value)
            lb, ub, tf = params_space[param_name]
            assert lb - EPS < value < ub + EPS, \
                f"Parameter {param_name} (={value}) is out of bound [{lb}, {ub}]."
            param_value = tf(value)
            if param_name.startswith("GP_"):
                params_to_update.setdefault("global_place_stages", [{}])
                subject = params_to_update["global_place_stages"][0]
                entry_name = param_name.lstrip("GP_")
            else:
                subject = params_to_update
                entry_name = param_name
            subject[entry_name] = param_value

        self.params.fromJson(params_to_update)


    def _genotype2phenotype(self, x):
        params_name = list(params_space.keys())
        x = dict(tuple(zip(params_name, list(x))))
        self._load_genotype(x)

        original_abort_signal_handler = signal.signal(signal.SIGABRT, abort_signal_handler)
        original_timeout_handler      = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(600) # maximum execution time
        t_start = time.time()
        try:
            with th.no_grad():
                self.dmp_plcr.pos[0].data.copy_(
                    th.from_numpy(self.dmp_plcr._initialize_position(self.params, self.dmp_pldb)).to(self.dmp_plcr.device) )
            
            gp_metrics = self.dmp_plcr(self.params, self.dmp_pldb)

            macro_pos = self.dmp_pldb.export(self.params, self.placedb.macro_lst)
            for node_name in macro_pos.keys():
                x = macro_pos[node_name][0] / (self.dmp_pldb.xh - self.dmp_pldb.xl) * self.placedb.canvas_width
                y = macro_pos[node_name][1] / (self.dmp_pldb.yh - self.dmp_pldb.yl) * self.placedb.canvas_height
                macro_pos[node_name] = (x, y)
            return macro_pos
        except KeyboardInterrupt:
            exit(0)
        except AbortSignalException:
            return {}
        except TimeoutError:
            return {}
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGABRT, original_abort_signal_handler)
            signal.signal(signal.SIGALRM, original_timeout_handler)


    def __deepcopy__(self, memo=None):
        return self
    