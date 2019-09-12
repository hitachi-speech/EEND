#!/bin/bash

# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Licensed under the MIT license.
#
# BLSTM-based model experiment
./run.sh --train-config conf/blstm/train.yaml --infer-config conf/blstm/infer.yaml $*
