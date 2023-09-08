import socket
from contextlib import contextmanager

from bev.config import SCPConfig
from paramiko.ssh_exception import AuthenticationException, NoValidConnectionsError, SSHException
from tarn import SCP
from tarn.location.scp import UnknownHostException


# TODO: move this to tarn and bev
#  this is a super ugly crutch, but it's only used in test, so ok for now
class SFTP(SCP):
    @contextmanager
    def _connect(self):
        try:
            self.ssh.connect(
                self.hostname, self.port, self.username, self.password, key_filename=self.key, auth_timeout=10
            )
        except AuthenticationException:
            raise AuthenticationException(self.hostname) from None
        except socket.gaierror:
            raise UnknownHostException(self.hostname) from None
        except (SSHException, NoValidConnectionsError):
            yield None
            return

        self.levels = 1, 63
        try:
            with self.ssh.open_sftp() as scp:

                class Wrapper:
                    @staticmethod
                    def get(*args, recursive=None):
                        return scp.get(*args)

                yield Wrapper

        finally:
            self.ssh.close()


class SFTPConfig(SCPConfig):
    def build(self):
        return SFTP(self.host, self.root)
