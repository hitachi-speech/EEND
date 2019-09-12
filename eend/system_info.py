# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Licensed under the MIT license.

import sys
import chainer
import cupy
import cupy.cuda
from cupy.cuda import cudnn


def print_system_info():
    pyver = sys.version.replace('\n', ' ')
    print(f"python version: {pyver}")
    print(f"chainer version: {chainer.__version__}")
    print(f"cupy version: {cupy.__version__}")
    print(f"cuda version: {cupy.cuda.runtime.runtimeGetVersion()}")
    print(f"cudnn version: {cudnn.getVersion()}")
