
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

import model_io
from dataloader import DepthDataLoader
from models import UnetAdaptiveBins
from utils import RunningAverageDict

class RunningAverage:
    def __init__(self):
        self.avg = 0
        self.count = 0

    def append(self, value):
        self.avg = (value + self.count * self.avg) / (self.count + 1)
        self.count += 1

    def get_value(self):
        return self.avg

class RunningAverageDict:
    def __init__(self):
        self._dict = None

    def update(self, new_dict):
        if self._dict is None:
            self._dict = dict()
            for key, value in new_dict.items():
                self._dict[key] = RunningAverage()

        for key, value in new_dict.items():
            self._dict[key].append(value)

    def get_value(self):
        return {key: value.get_value() for key, value in self._dict.items()}

def compute_errors(ground_truth, pred):

    # threshold accuracy, the % of y_p such that max((y_p/y^_p), (y^_p/y_p)) < threshold 
    a1 = np.mean(np.maximum((ground_truth / pred), (pred / ground_truth)) < 1.25)
    a2 = np.mean(np.maximum((ground_truth / pred), (pred / ground_truth)) < 1.25 ** 2)
    a3 = np.mean(np.maximum((ground_truth / pred), (pred / ground_truth)) < 1.25 ** 3)

    # average relative error
    abs_rel = np.mean(np.abs(ground_truth - pred) / ground_truth)
    # square relative error
    sq_rel = np.mean(((ground_truth - pred) ** 2) / ground_truth)

    # root mean squared error
    rmse = (ground_truth - pred) ** 2
    rmse = np.sqrt(np.mean(rmse))

    # root mean squared log error
    rmse_log = (np.log(ground_truth) - np.log(pred)) ** 2
    rmse_log = np.sqrt(np.mean(rmse_log))

    # silog loss
    err = np.log(pred) - np.log(ground_truth)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    # average log_10 error
    log_10 = (np.abs(np.log10(ground_truth) - np.log10(pred))).mean()

    # a1, a2, a3, abs_rel, rmse, log_10 for nyu
    return dict(a1=a1, a2=a2, a3=a3, abs_rel=abs_rel, rmse=rmse, log_10=log_10, rmse_log=rmse_log,
                silog=silog, sq_rel=sq_rel)


def evaluate(pred, gt, metrics):
    
    evaluation = compute_errors(gt, pred)
    metrics.update(evaluation)














