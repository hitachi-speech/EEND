#!/usr/bin/env python3

# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Licensed under the MIT license.

import os
import chainer
import subprocess
import cupy


def get_free_gpus():
    """ Get IDs of free GPUs using `nvidia-smi`.

    Returns:
        sorted list of GPUs which have no running process.
    """
    p = subprocess.Popen(
            ["nvidia-smi",
             "--query-gpu=index,gpu_bus_id",
             "--format=csv,noheader"],
            stdout=subprocess.PIPE)
    stdout, stderr = p.communicate()
    gpus = {}
    for line in stdout.decode('utf-8').strip().split(os.linesep):
        idx, busid = line.strip().split(',')
        gpus[busid] = int(idx)
    p = subprocess.Popen(
            ["nvidia-smi",
             "--query-compute-apps=pid,gpu_bus_id",
             "--format=csv,noheader"],
            stdout=subprocess.PIPE)
    stdout, stderr = p.communicate()
    for line in stdout.decode('utf-8').strip().split(os.linesep):
        pid, busid = line.strip().split(',')
        del gpus[busid]
    return sorted([gpus[busid] for busid in gpus])


def use_single_gpu():
    """ Use single GPU device.

    If CUDA_VISIBLE_DEVICES is set, select a device from the variable.
    Otherwise, get a free GPU device and use it.

    Returns:
        assigned GPU id.
    """
    cvd = os.environ.get('CUDA_VISIBLE_DEVICES')
    if cvd is None:
        # no GPUs are researved
        cvd = get_free_gpus()[0]
    elif ',' in cvd:
        # multiple GPUs are researved
        cvd = int(cvd.split(',')[0])
    else:
        # single GPU is reserved
        cvd = int(cvd)
    # Use the GPU immediately
    chainer.cuda.get_device_from_id(cvd).use()
    cupy.empty((1,), dtype=cupy.float32)
    return cvd
