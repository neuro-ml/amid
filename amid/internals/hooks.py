from bev.hookspecs import hookimpl

from .yandex import YandexDisk  # noqa: F401


@hookimpl
def register_config_extensions():
    pass
