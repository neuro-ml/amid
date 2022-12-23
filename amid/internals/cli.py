from argparse import ArgumentParser
from functools import partial
from pathlib import Path

from bev.cli.init import init

from .registry import gather_datasets


def main():
    parser = ArgumentParser()
    subparsers = parser.add_subparsers()

    new = subparsers.add_parser('init')
    new.set_defaults(callback=partial(init, Path(__file__).resolve().parent.parent / 'data'))
    new.add_argument('-p', '--permissions')
    new.add_argument('-g', '--group')

    new = subparsers.add_parser('populate')
    new.set_defaults(callback=populate)
    new.add_argument('dataset', help='the dataset name')
    new.add_argument('root', help='raw data location')
    new.add_argument('--ignore-errors', action='store_true', default=False)
    new.add_argument('--fetch', action='store_true', default=False)
    new.add_argument('--n-jobs', type=int, default=1)

    args = vars(parser.parse_args())
    if 'callback' not in args:
        parser.print_help()
    else:
        callback = args.pop('callback')
        callback(**args)


def populate(dataset, root, ignore_errors, n_jobs, fetch):
    cls = gather_datasets()[dataset][0]
    ds = cls(root=root)
    success, errors = ds._populate(n_jobs=n_jobs, fetch=fetch, ignore_errors=ignore_errors)
    print(f'Total added: {success} entries, and encountered {errors} errors')
