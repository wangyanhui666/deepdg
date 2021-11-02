# coding=utf-8
from alg.algs.ERM import ERM
from alg.algs.MMD import MMD
from alg.algs.CORAL import CORAL
from alg.algs.DANN import DANN
from alg.algs.DANN_RES import DANN_RES_C, DANN_RES_A
from alg.algs.DAAN import DAAN
from alg.algs.RSC import RSC
from alg.algs.Mixup import Mixup
from alg.algs.MLDG import MLDG
from alg.algs.GroupDRO import GroupDRO
from alg.algs.ANDMask import ANDMask

ALGORITHMS = [
    'ERM',
    'Mixup',
    'CORAL',
    'MMD',
    'DANN',
    'DAAN',
    'DANN_RES_C',
    'DANN_RES_A',
    'MLDG',
    'GroupDRO',
    'RSC',
    'ANDMask'
]


def get_algorithm_class(algorithm_name):
    if algorithm_name not in globals():
        raise NotImplementedError(
            "Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]
