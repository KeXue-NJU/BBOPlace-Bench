import os
import copy
import signal
import numpy as np
import torch as th

from utils.read_benchmark.read_aux import write_pl
from utils.read_benchmark.read_def import write_def
from utils.signal_handler import abort_signal_handler
from utils.debug import *
from utils.constant import INF

from thirdparty.dreamplace.Params import Params as DMPParams
from thirdparty.dreamplace.PlaceDB import PlaceDB as DMPPlaceDB
from thirdparty.dreamplace.NonLinearPlace import NonLinearPlace
import thirdparty.dreamplace.configure as configure
import thirdparty.dreamplace.ops.place_io.place_io as place_io

from PIL import Image


class GPEvaluator:
    DMP_CONFIG_PATH = "config/algorithm/dmp_config"
    DMP_TEMP_BENCHMARK_PATH = "benchmarks/.tmp"

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

    def __init__(self, args, placedb) -> None:
        self.args = args
        self.placedb = placedb

        self.dmp_params = DMPParams()
        self.dmp_placedb = None

    
        # load default dreamplace config
        self._load_dmp_config()
        os.environ["OMP_NUM_THREADS"] = str(self.dmp_params.num_threads)
        assert (not self.dmp_params.gpu) or configure.compile_configurations["CUDA_FOUND"] == 'TRUE', \
            "CANNOT enable GPU without CUDA compiled"

        # prepare benchmark
        self._prepare_benchmark()

        # prepare placement, figure database
        self.empty_saving_data()

        self.n_eval = 0


    def evaluate(self, macro_pos):
        if len(macro_pos) == 0:
            return INF
        
        if self.dmp_placedb is None:
            self._init_dmp(macro_pos=macro_pos)

        original_abort_signal_handler = signal.signal(signal.SIGABRT, abort_signal_handler)
        try:
            self._update_dmp_placedb(macro_pos=macro_pos)
            self._update_dmp_placer()

            gp_hpwl = self.placer(self.dmp_params, self.dmp_placedb)[-1].hpwl.cpu().item()
            
            self.saving_data["placement"][gp_hpwl] = (self.dmp_placedb.node_x.copy(),
                                                        self.dmp_placedb.node_y.copy())
            self.saving_data["figure"][gp_hpwl] = copy.copy(self.placer.pos[0].data.clone().cpu().numpy())
        except KeyboardInterrupt:
            exit(0)
        except:
            return INF
        finally:
            signal.signal(signal.SIGABRT, original_abort_signal_handler)

        return gp_hpwl
    
    def _init_dmp(self, macro_pos):
        self._prepare_placement_file(macro_pos=macro_pos)
        self.dmp_placedb = DMPPlaceDB()
        self.dmp_placedb(self.dmp_params)

        self.dmp_node_names = self.dmp_placedb.node_names.astype(np.str_)
        mask = np.char.find(self.dmp_node_names, "DREAMPlace") != -1
        modified_names = np.char.split(self.dmp_node_names[mask], '.').tolist()
        self.dmp_node_names[mask] = [name[0] for name in modified_names]
        self.placer = NonLinearPlace(self.dmp_params, self.dmp_placedb, timer=None)
    
    
    def _load_dmp_config(self):
        ROOT_DIR = self.args.ROOT_DIR
        json_file = f"{self.args.benchmark}.json"
        json_path = os.path.join(
            ROOT_DIR,
            GPEvaluator.DMP_CONFIG_PATH,
            json_file
        )
        self.dmp_params.load(json_path)
        self.dmp_params.benchmark = self.args.benchmark
        self.dmp_params.random_center_init_flag = 1


    def _prepare_benchmark(self):
        os.makedirs(self._temp_benchmark_path, exist_ok=True)
        type_mapping = {
            "aux": self._prepare_benchmark_aux,
            "def": self._prepare_benchmark_def,
        }
        if self.args.benchmark_type in type_mapping:
            type_mapping[self.args.benchmark_type]()
        else:
            raise NotImplementedError
        
    def _prepare_benchmark_aux(self):
        self._link_files(GPEvaluator.AUX_FILES)

        suffix2path = \
            lambda suffix: os.path.join(
                self._temp_benchmark_path,
                "%(benchmark)s" % self.args.__dict__
            ) + suffix
        self.dmp_params.fromJson(
            {
                "aux_input": suffix2path(".aux")
            }
        )
        
    def _prepare_benchmark_def(self):
        self._link_files(GPEvaluator.DEF_FILES)

        suffix2path = \
            lambda suffix: os.path.join(
                self._temp_benchmark_path,
                "%(benchmark)s" % self.args.__dict__
            ) + suffix
        self.dmp_params.fromJson(
            {
                "def_input": suffix2path(".def"),
                "lef_input": suffix2path(".lef"),
                "verilog_input": suffix2path(".v"),
                "early_lib_input": suffix2path("_Early.lib"),
                "late_lib_input": suffix2path("_Late.lib"),
                "sdc_input": suffix2path(".sdc")
            }
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

    def _prepare_placement_file(self, macro_pos):
        file_name = os.path.join(self._temp_benchmark_path, self.args.benchmark)
        type_mapping = {
            "aux" : lambda macro_pos, placedb : write_pl(file_name+".pl", macro_pos=macro_pos, placedb=placedb),
            "def" : lambda macro_pos, placedb : write_def(file_name+".def", macro_pos=macro_pos, placedb=placedb),
        }
        if self.args.benchmark_type in type_mapping:
            type_mapping[self.args.benchmark_type](macro_pos=macro_pos, placedb=self.placedb)
        else:
            raise NotImplementedError
        
    def empty_saving_data(self):
        self.saving_data = {
            "placement" : {},
            "figure" : {}
        }

    def _update_dmp_placedb(self, macro_pos):
        if self.args.benchmark_type == "aux":
            dmp_scale_factor_x, dmp_scale_factor_y = 1, 1
        else:
            dmp_scale_factor_x, dmp_scale_factor_y = self.dmp_placedb.xh/self.placedb.canvas_width, \
                                                     self.dmp_placedb.yh/self.placedb.canvas_height

        for macro in macro_pos:
            index = np.where(self.dmp_node_names == macro)
            pos_x = round(macro_pos[macro][0] * dmp_scale_factor_x) 
            pos_y = round(macro_pos[macro][1] * dmp_scale_factor_y) 
            self.dmp_placedb.node_x[index] = pos_x
            self.dmp_placedb.node_y[index] = pos_y
        
        node_x, node_y = self.dmp_placedb.unscale_pl(self.dmp_params.shift_factor, 
                                                     self.dmp_params.scale_factor)
        place_io.PlaceIOFunction.apply(self.dmp_placedb.rawdb, node_x, node_y, all_movable=True)
    
    def _update_dmp_placer(self):
        with th.no_grad():
            self.placer.pos[0].data.copy_(
                th.from_numpy(self.placer._initialize_position(self.dmp_params, self.dmp_placedb)).to(self.placer.device) )
            

    def save_placement(self, hpwl, placement_name):
        self.dmp_placedb.node_x[:] = self.saving_data["placement"][hpwl][0].copy()
        self.dmp_placedb.node_y[:] = self.saving_data["placement"][hpwl][1].copy()
        # unscale locations
        node_x, node_y = self.dmp_placedb.unscale_pl(self.dmp_params.shift_factor, 
                                                     self.dmp_params.scale_factor)
        # update raw database
        place_io.PlaceIOFunction.apply(self.dmp_placedb.rawdb, node_x, node_y, all_movable=True)

        self.dmp_placedb.write(
            self.dmp_params, 
            placement_name
        )
    
    def __deepcopy__(self, memo=None):
        return self
    
    def plot(self, hpwl, figure_name):
        pos = self.saving_data["figure"][hpwl]
        self.placer.plot(
            self.dmp_params,
            None,
            None,
            pos,
            figure_name, 
        )

        img = Image.open(figure_name)
        out = img.transpose(Image.FLIP_TOP_BOTTOM)
        img.close()
        out.save(figure_name)

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
            GPEvaluator.DMP_TEMP_BENCHMARK_PATH,
            "%(benchmark)s_%(unique_token)s" % self.args.__dict__
        )



    
        
    

