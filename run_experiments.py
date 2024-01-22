import os
import sys
import subprocess
import argparse
import multiprocessing
from itertools import product


def execute(cmd):
    os.system(command=cmd)


def parse_cmd(cmd):
    return cmd + '.py'


def parse_args_grid(cmd, args):
    arg_dict = {}
    parallel = False
    for index, arg in enumerate(args):
        if '--' in arg:
            if arg == '--parallel':
                parallel = True
            else:
                arg_key = arg
                arg_dict[arg_key] = []
        else:
            arg_dict[arg_key].append(arg)
    print(arg_dict)
    # 参数空间的笛卡尔积
    grid_exp_list = [dict(zip(arg_dict, v)) for v in product(*arg_dict.values())]
    print(grid_exp_list)
    if parallel:
        # 终端并行
        for exp in grid_exp_list:
            command = 'python' + ' ' + cmd + ' '
            for key, value in exp.items():
                command += key + ' ' + value + ' '
            print(command)
            p = multiprocessing.Process(target=execute, kwargs={'cmd': command})
            p.start()
    else:
        # 终端串行
        for exp in grid_exp_list:
            command = 'python' + ' ' + cmd + ' '
            for key, value in exp.items():
                command += key + ' ' + value + ' '
            print(command)
            execute(command)


if __name__ == '__main__':
    cmd = sys.argv[1]
    cmd = parse_cmd(cmd)
    args = sys.argv[2:]
    parse_args_grid(cmd, args)