from bev.config import SCPConfig
from tarn import SFTP


# TODO: move this to bev
class SFTPConfig(SCPConfig):
    def build(self):
        return SFTP(self.host, self.root)
