from pathlib import Path

import typer
from bev.cli.app import command
from bev.cli.init import init as bev_init

from .registry import gather_datasets


# older versions of typer don't have this arg, and it's simpler to avoid conflicts this way
try:
    app = typer.Typer(pretty_exceptions_show_locals=False)
except TypeError:
    app = typer.Typer()


def main():
    app()


@command(app)
def init(
    permissions: str = typer.Option(
        None,
        '--permissions',
        '-p',
        help='The permissions mask used to create the storage, e.g. 770',
    ),
    group: str = typer.Option(
        None,
        '--group',
        '-g',
        help='The group used to create the storage',
    ),
):
    return bev_init(Path(__file__).resolve().parent.parent / 'data', permissions, group)


@app.command()
def populate(
    dataset: str = typer.Argument(..., help='The dataset to populate'),
    root: Path = typer.Argument(..., help='The path to the downloaded raw data'),
    ignore_errors: bool = typer.Option(
        False,
        help='Whether to ignore all the exception during population. '
        'Warning! The failed ids will be excluded from the populated dataset',
    ),
    n_jobs: int = typer.Option(1, help='How many threads to use for population'),
    fetch: bool = typer.Option(False, help='Whether to fetch the missing data from remote locations, if any'),
):
    try:
        cls = gather_datasets()[dataset][0]
        ds = cls(root=root)
        success, errors = ds._populate(n_jobs=n_jobs, fetch=fetch, ignore_errors=ignore_errors)
        print(f'Total added: {success} entries, and encountered {errors} errors')
    except KeyboardInterrupt as e:
        raise RuntimeError from e
