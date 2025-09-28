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
import socket
import signal
import time
import tempfile
import json
import subprocess
import multiprocessing as mp
import select

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
    "GP_num_bins_x": (0, 2, sel_func([1024, 2048])),
    "GP_num_bins_y": (0, 2, sel_func([1024, 2048])),
    "GP_optimizer": (0, 2, sel_func(["adam", "nesterov"])),
    "GP_wirelength": (0, 2, sel_func(["weighted_average", "logsumexp"])),
    "GP_iteration": (0, 1, sel_func([1000])),

# uniform
    "GP_Llambda_density_weight_iteration": (1, 3, round_func),
    "GP_Lsub_iteration": (1, 3, round_func),
    "GP_learning_rate": (0.001, 0.01, orig_func),
    "GP_learning_rate_decay": (0.99, 1.0, orig_func),
    "RePlAce_LOWER_PCOF": (0.9, 0.99, orig_func),
    "RePlAce_UPPER_PCOF": (1.02, 1.15, orig_func),
    "RePlAce_ref_hpwl": (150000, 550000, round_func),
    "density_weight": (1e-06, 1e-04, orig_func),
    "gamma": (1, 4, orig_func),
    "stop_overflow": (0.06, 0.1, orig_func),
    "target_density": (0.8, 1.2, orig_func),

}

class DMPPlacer(BasicPlacer):
    DMP_CONFIG_PATH = "config/algorithm/dmp_config"
    DMP_TEMP_BENCHMARK_PATH = "benchmarks/.tmp/HPO"
    DMP_RESULT_DIR = os.path.join(
        "results",
        "%(name)s",
        "%(benchmark)s",
        "%(unique_token)s",
        "dmp_results"
    )
    SOCK_PATH = os.path.join(
        "sock_path",
        "dmp_worker_HPO_%(unique_token)s.sock"
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

        self._worker = None
        self._sock = None
        self._sock_path = None
        self._worker_inited = False
        self.worker_path = os.path.join(self.args.SOURCE_DIR, "placer/dmp_worker.py")
        
        self.params = DMPParams()
        self._load_dmp_config()
        self._prepare_benchmark()

        self.timeout_seconds = args.timeout_seconds

        self._sock_path = os.path.join(self.args.ROOT_DIR, DMPPlacer.SOCK_PATH % self.args.__dict__)
        os.makedirs(os.path.dirname(self._sock_path), exist_ok=True)

        # self.dmp_pldb = DMPPlaceDB()

        # # load default dreamplace config
        # self._load_dmp_config()

        # # prepare benchmark
        # self._prepare_benchmark()

        # self.params.fromJson(
        #     {
        #         "plot_flag": 0,
        #         "timing_opt_flag": 0,
        #         "random_seed": self.args.seed,
        #         "result_dir": self._result_dir,
        #         "random_center_init_flag": 1,
        #     }
        # )

        # self.dmp_pldb(self.params)

        # self.dmp_plcr = NonLinearPlace(self.params, self.dmp_pldb, timer=None)


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
        
        if not self._ensure_worker():
            return {}

        # place request
        req = {
            "cmd": "place",
            "params_update": {},
            "macro_lst": self.placedb.macro_lst,
        }
        deadline = time.time() + self.timeout_seconds
        try:
            self._sock.sendall((json.dumps(req) + "\n").encode())
            remaining = deadline - time.time()
            if remaining <= 0:
                self._cleanup_worker()
                return {}
            r, _, _ = select.select([self._sock], [], [], remaining)
            if not r:
                self._cleanup_worker()
                return {}
            
            data = b""
            while True:
                remaining = deadline - time.time()
                if remaining <= 0:
                    self._cleanup_worker()
                    return {}
                    
                chunk = self._sock.recv(65536)
                if not chunk:
                    break
                data += chunk
                if b"\n" in chunk:
                    break
            text = data.decode(errors="ignore")
            lines = [ln for ln in text.splitlines() if ln.strip()]
            last_line = lines[-1] if lines else ""
            out = json.loads(last_line) if last_line else {"ok": False}
        except:
            self._cleanup_worker()
            return {}

        if isinstance(out, dict) and out.get("ok") and \
           isinstance(out.get("macro_pos"), dict):
            macro_pos = out["macro_pos"]
            for k, v in list(macro_pos.items()):
                try:
                    macro_pos[k] = (v[0], v[1])
                except:
                    macro_pos[k] = tuple(v)
            return macro_pos
        else:
            self._cleanup_worker()
            return {}

    def _cleanup_worker(self):
        if self._sock is not None:
            self._sock.close()
            self._sock = None
        if self._worker is not None:
            self._worker.kill()
            self._worker = None
        if self._sock_path is not None and os.path.exists(self._sock_path):
            os.unlink(self._sock_path)
        self._worker_inited = False

    def _ensure_worker(self):
        if self._worker is not None and \
           self._worker.poll() is None and \
           self._sock is not None:
            return True
        
        if os.path.exists(self._sock_path):
            os.unlink(self._sock_path)
        
        # init worker
        try:
            self._worker = subprocess.Popen(
                ["python3", self.worker_path, "--sock", self._sock_path],
                stdin=None,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=False,
                cwd=self.args.ROOT_DIR,
            )
        except Exception:
            self._cleanup_worker()
            return False

        # init sock
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        deadline = time.time() + min(10, self.timeout_seconds)
        connected = False
        while time.time() < deadline:
            try:
                self._sock.connect(self._sock_path)
                connected = True
                break
            except Exception:
                time.sleep(0.05)
        if not connected:
            self._cleanup_worker()
            return False
        self._sock.settimeout(self.timeout_seconds)

        # init worker
        init_msg = json.dumps({
            "cmd": "init",
            "args": {
                "ROOT_DIR": self.args.ROOT_DIR,
                "temp_subdir": "HPO",
                "name": self.args.name,
                "benchmark": self.args.benchmark,
                "benchmark_type": self.args.benchmark_type,
                "unique_token": self.args.unique_token,
                "seed": self.args.seed,
            },
            "canvas_width": self.placedb.canvas_width,
            "canvas_height": self.placedb.canvas_height,
        }) + "\n"
        try:
            self._sock.sendall(init_msg.encode())
            data = b""
            while True:
                chunk = self._sock.recv(65536)
                if not chunk:
                    break
                data += chunk
                if b"\n" in chunk:
                    break
            line = data.decode(errors="ignore").strip()
            ack_obj = json.loads(line) if line else {"ok": False}
        except:
            self._cleanup_worker()
            return False
        if not ack_obj.get("ok"):
            self._cleanup_worker()
            return False
        self._worker_inited = True
        return True

    def __deepcopy__(self, memo=None):
        return self
    
    