from argparse import ArgumentParser
from functools import partial
from pathlib import Path

from bev.cli.init import init


def main():
    parser = ArgumentParser()
    subparsers = parser.add_subparsers()

    new = subparsers.add_parser('init')
    new.set_defaults(callback=partial(init, Path(__file__).resolve().parent.parent / 'data'))
    new.add_argument('-p', '--permissions')
    new.add_argument('-g', '--group')

    args = vars(parser.parse_args())
    if 'callback' not in args:
        parser.print_help()
    else:
        callback = args.pop('callback')
        callback(**args)
