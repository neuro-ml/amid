from bev import Repository


def get_repo(path=None, strict=True):
    try:
        repo = Repository.from_here('../data')
    except FileNotFoundError:
        if strict:
            raise
        return

    if path:
        repo = repo / path
    return repo
