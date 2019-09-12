#!/usr/bin/env python3

# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Licensed under the MIT license.

import argparse
import h5py
import numpy as np
import os
from scipy.signal import medfilt

parser = argparse.ArgumentParser(description='make rttm from decoded result')
parser.add_argument('file_list_hdf5')
parser.add_argument('out_rttm_file')
parser.add_argument('--threshold', default=0.5, type=float)
parser.add_argument('--frame_shift', default=256, type=int)
parser.add_argument('--subsampling', default=1, type=int)
parser.add_argument('--median', default=1, type=int)
parser.add_argument('--sampling_rate', default=16000, type=int)
args = parser.parse_args()

filepaths = [line.strip() for line in open(args.file_list_hdf5)]
filepaths.sort()

with open(args.out_rttm_file, 'w') as wf:
    for filepath in filepaths:
        session, _ = os.path.splitext(os.path.basename(filepath))
        data = h5py.File(filepath, 'r')
        a = np.where(data['T_hat'][:] > args.threshold, 1, 0)
        if args.median > 1:
            a = medfilt(a, (args.median, 1))
        for spkid, frames in enumerate(a.T):
            frames = np.pad(frames, (1, 1), 'constant')
            changes, = np.where(np.diff(frames, axis=0) != 0)
            fmt = "SPEAKER {:s} 1 {:7.2f} {:7.2f} <NA> <NA> {:s} <NA>"
            for s, e in zip(changes[::2], changes[1::2]):
                print(fmt.format(
                      session,
                      s * args.frame_shift * args.subsampling / args.sampling_rate,
                      (e - s) * args.frame_shift * args.subsampling / args.sampling_rate,
                      session + "_" + str(spkid)), file=wf)
