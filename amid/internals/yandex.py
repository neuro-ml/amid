import io
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Sequence, Tuple, Union
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve

import requests
from bev.config import RemoteConfig, register
from tarn import RemoteStorage
from tarn.config import HashConfig, StorageConfig
from tarn.digest import key_to_relative
from tarn.utils import Key
from yaml import safe_load


class YandexDisk(RemoteStorage):
    def __init__(self, url: str, optional: bool):
        self.optional = optional
        self.url = url
        self._api_url = 'https://cloud-api.yandex.net/v1/disk/public/resources/'
        self.levels = self.hash = None

    def fetch(
        self, keys: Sequence[Key], store: Callable[[Key, Path], Any], config: HashConfig
    ) -> Sequence[Tuple[Any, bool]]:
        results = []
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / 'source'
            if keys and not self._get_config(config):
                return [(None, False)] * len(keys)

            for key in keys:
                try:
                    self._fetch_tree(key_to_relative(key, self.levels), source)

                    value = store(key, source)
                    shutil.rmtree(source)
                    results.append((value, True))

                except requests.exceptions.ConnectionError:
                    results.append((None, False))

                shutil.rmtree(source, ignore_errors=True)

        return results

    def _get_config(self, reference):
        try:
            if self.levels is None:
                response = requests.get(self._api_url, dict(public_key=self.url, path='/config.yml')).json()
                if 'error' in response:
                    raise requests.exceptions.ConnectionError(f'Error while downloading: {response["error"]}')

                url = response['file']
                with io.StringIO(requests.get(url).text) as stream:
                    config = StorageConfig.parse_obj(safe_load(stream))
                self.hash, self.levels = config.hash, config.levels

            assert self.hash == reference, (self.hash, reference)
            return True

        except requests.exceptions.ConnectionError:
            if not self.optional:
                raise

            return False

    @staticmethod
    def _fetch_file(url, local):
        try:
            urlretrieve(url, str(local))
        except (HTTPError, URLError) as e:
            raise requests.exceptions.ConnectionError from e

    def _fetch_tree(self, relative, local):
        local.mkdir(parents=True, exist_ok=True)

        count, total = 0, 1
        while count < total:
            content = requests.get(
                self._api_url, dict(public_key=self.url, path=str('/' / relative), offset=count)
            ).json()
            if 'error' in content:
                raise requests.exceptions.ConnectionError(f'Error while downloading: {content["error"]}')

            content = content['_embedded']
            total = content['total']
            count += len(content['items'])

            for entry in content['items']:
                name = entry['name']
                if 'file' in entry:
                    self._fetch_file(entry['file'], local / name)
                else:
                    self._fetch_tree(relative / name, local / name)


@register('ya-disk')
class YandexDiskConfig(RemoteConfig):
    url: str

    @classmethod
    def from_string(cls, v):
        return cls(url=v)

    def build(self, root: Union[Path, None], optional: bool) -> Union[RemoteStorage, None]:
        return YandexDisk(self.url, optional)
