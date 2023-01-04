from bev import Repository
from bev.exceptions import ConfigError


def get_repo(path=None, strict=True):
    try:
        repo = Repository.from_here('../data')
    except (ConfigError, FileNotFoundError):
        if strict:
            raise
        return

    if path:
        repo = repo / path
    return repo
