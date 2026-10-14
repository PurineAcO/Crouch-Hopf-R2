import numpy as np

class node:
    """节点类"""
    def __init__(self):
        x = 0.0
        y = 0.0

class cell:
    """单元类"""
    def __init__(self):
        index = 0      # 网格编号应从1开始
        center = (0.0,0.0)
        vol = 0.0
        nodelist : list[node] = [None,None,None,None]

        north = None
        south = None
        east = None
        west = None
        blk = 0         # 0 表示未分配的块

        rho = 0.0
        u = 0.0
        v = 0.0
        T = 0.0
        miubl = 0.0

        jacobi : list[np.ndarray] = [          None,
                                          None,None,None,
                                     None,None,None,None,None,
                                          None,None,None,
                                               None          ]


def at(place):
    if place=='c' : return 6
    elif place=='e' : return 7
    elif place=='w' : return 5
    elif place=='n' : return 2
    elif place=='s' : return 10
    elif place=='nn' : return 0
    elif place=='ss' : return 12
    elif place=='ww' : return 4
    elif place=='ee' : return 8
    elif place=='ne' : return 3
    elif place=='nw' : return 1
    elif place=='sw' : return 9
    elif place=='se' : return 11
    else : raise ValueError("Not Correct Place Token: ",place)