#!/usr/bin/env python3

# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Licensed under the MIT license.

import argparse
import yaml


def print_var_assign_statements(obj, prefix=''):
    """ Print variable assignment statements from yaml object.
        - { key: value } -> key=value
        - { parent: { child: value } } -> parent_child=value
        - { key: [ val1, val2 ]} -> key_0=val1 key_1=val2
    """
    if isinstance(obj, dict):
        for key in obj:
            child_prefix = prefix + '_' + key if prefix else key
            print_var_assign_statements(obj[key], child_prefix)
    elif isinstance(obj, list):
        for key, val in enumerate(obj):
            child_prefix = prefix + '_' + str(key) if prefix else str(key)
            print_var_assign_statements(val, child_prefix)
    else:
        if obj is None:
            obj = ''
        elif obj is False:
            obj = 'false'
        elif obj is True:
            obj = 'true'
        print(f"{prefix}={obj}")


parser = argparse.ArgumentParser()
parser.add_argument('input_yaml')
parser.add_argument('--prefix', default='')
args = parser.parse_args()

data = yaml.safe_load(open(args.input_yaml))
print_var_assign_statements(data, prefix=args.prefix)
