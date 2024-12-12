import math
import numpy as np
import logging

from dataclasses import dataclass
from .basic_placer import BasicPlacer
from utils.constant import INF
from utils.debug import *

class BstarTreePlacer(BasicPlacer):
    def __init__(self, args, placedb) -> None:
        super(BstarTreePlacer, self).__init__(args=args, placedb=placedb)

        self.node = Node

        self.sum_size_x = 0
        for macro in self.placedb.node_info:
            self.sum_size_x += self.placedb.node_info[macro]["size_x"]

    
    def _genotype2phenotype(self, x:dict):
        def dfs_update(root, macro_pos):
            root_macro = self.placedb.macro_lst[root.id]
            root_pos_x, root_pos_y = macro_pos[root_macro]
            root_size_x, root_size_y = self.placedb.node_info[root_macro]["size_x"], self.placedb.node_info[root_macro]["size_y"]

            left_child_id = root.left
            if left_child_id is not None:
                left_child_macro = self.placedb.macro_lst[left_child_id]
                left_pos_x = root_pos_x + root_size_x
                left_pos_y = np.max(canvas_macro_height[left_pos_x : left_pos_x + self.placedb.node_info[left_child_macro]["size_x"]])
                macro_pos[left_child_macro] = (left_pos_x, left_pos_y)

                # update canvas_macro_height
                canvas_macro_height[left_pos_x : left_pos_x + self.placedb.node_info[left_child_macro]["size_x"]] = left_pos_y + self.placedb.node_info[left_child_macro]["size_y"]

                left_child = x[left_child_id]
                dfs_update(left_child, macro_pos)

            right_child_id = root.right
            if right_child_id is not None:
                right_child_macro = self.placedb.macro_lst[right_child_id]
                right_pos_x = root_pos_x
                right_pos_y = np.max(canvas_macro_height[right_pos_x : right_pos_x + self.placedb.node_info[right_child_macro]["size_x"]])
                macro_pos[right_child_macro] = (right_pos_x, right_pos_y)

                # update canvas_macro_height
                canvas_macro_height[right_pos_x : right_pos_x + self.placedb.node_info[right_child_macro]["size_x"]] = right_pos_y + self.placedb.node_info[right_child_macro]["size_y"]

                right_child = x[right_child_id]
                dfs_update(right_child, macro_pos)


        macro_pos = {}
        x = x[0]
        root_id = x["root"]
        root = x[root_id]
        root_macro = self.placedb.macro_lst[root_id]
        macro_pos[root_macro] = (0, 0)

        # determine the y posiiton of macro
        canvas_macro_height = np.zeros(shape=self.sum_size_x)
        canvas_macro_height[0:self.placedb.node_info[root_macro]["size_x"]] += self.placedb.node_info[root_macro]["size_y"]
        dfs_update(root, macro_pos)

        return macro_pos

@dataclass
class Node:
    id : int
    parent : int
    left : int
    right : int


    
