#!/usr/bin/env python3

# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Licensed under the MIT license.
#
# averaging chainer serialized models

import numpy as np
import argparse


def average_model_chainer(ifiles, ofile):
    omodel = {}
    # get keys from the first file
    model = np.load(ifiles[0])
    for x in model:
        if 'model' in x:
            print(x)
    keys = [x.split('main/')[1] for x in model if 'model' in x]
    print(keys)
    for path in ifiles:
        model = np.load(path)
        for key in keys:
            val = model['updater/model:main/{}'.format(key)]
            if key not in omodel:
                omodel[key] = val
            else:
                omodel[key] += val
    for key in keys:
        omodel[key] /= len(ifiles)
    np.savez_compressed(ofile, **omodel)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("ofile")
    parser.add_argument("ifiles", nargs='+')
    parser.add_argument("--backend", default='chainer',
                        choices=['chainer', 'pytorch'])
    args = parser.parse_args()
    if args.backend == 'chainer':
        average_model_chainer(args.ifiles, args.ofile)
    elif args.backend == 'pytorch':
        # TODO
        # average_model_pytorch(args.ifiles, args.ofile)
        raise NotImplementedError()
    else:
        raise ValueError()
