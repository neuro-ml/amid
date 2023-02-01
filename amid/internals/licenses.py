from typing import NamedTuple


class License(NamedTuple):
    name: str
    url: str


CC0_10 = License(name='CC0 1.0', url='https://creativecommons.org/publicdomain/zero/1.0/')
CC_BY_30 = License(
    name='CC BY 3.0',
    url='https://creativecommons.org/licenses/by/3.0/',
)
CC_BY_40 = License(
    name='CC BY 4.0',
    url='https://creativecommons.org/licenses/by/4.0/',
)
CC_BYNC_40 = License(
    name='CC BY-NC 4.0',
    url='https://creativecommons.org/licenses/by-nc/4.0/',
)
CC_BYND_40 = License(
    name='CC BY-ND 4.0',
    url='https://creativecommons.org/licenses/by-nd/4.0/',
)
CC_BYNCND_40 = License(
    name='CC BY-NC-ND 4.0',
    url='https://creativecommons.org/licenses/by-nc-nd/4.0/',
)
CC_BYSA_40 = License(
    name='CC BY-SA 4.0',
    url='https://creativecommons.org/licenses/by-sa/4.0/',
)
CC_BYNCSA_40 = License(
    name='CC BY-NC-SA 4.0',
    url='https://creativecommons.org/licenses/by-nc-sa/4.0/',
)

PhysioNet_RHD_150 = License(
    name='PhysioNet Restricted Health Data License 1.5.0',
    url='https://www.physionet.org/about/licenses/physionet-restricted-health-data-license-150/',
)

StanfordDSResearch = License(
    name='Stanford University Dataset Research Use Agreement',
    url='https://stanfordaimi.azurewebsites.net/datasets/e8ca74dc-8dd4-4340-815a-60b41f6cb2aa',  # TODO: separate link
)
