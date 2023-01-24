[![docs](https://img.shields.io/badge/-docs-success)](https://neuro-ml.github.io/amid/)
[![contribute](https://img.shields.io/badge/-contribute-success)](https://neuro-ml.github.io/amid/latest/CONTRIBUTING/)
[![pypi](https://img.shields.io/pypi/v/amid?logo=pypi&label=PyPi)](https://pypi.org/project/amid/)
![License](https://img.shields.io/github/license/neuro-ml/amid)

Awesome Medical Imaging Datasets (AMID) - a curated list of medical imaging datasets with unified interfaces

# Getting started

Just import a dataset and start using it!

Note that for some datasets you must manually download the raw files first.

```python
from amid.verse import VerSe

ds = VerSe()
# get the available ids
print(len(ds.ids))
i = ds.ids[0]

# use the available methods:
#   load the image and vertebrae masks
x, y = ds.image(i), ds.masks(i)
print(ds.split(i), ds.patient(i))

# or get a namedTuple-like object:
entry = ds(i)
x, y = entry.image, entry.masks
print(entry.split, entry.patient)
```

# Available datasets

| Name                                                                                                                                         |   Entries | Body region                         | Modality                        |
|:---------------------------------------------------------------------------------------------------------------------------------------------|----------:|:------------------------------------|:--------------------------------|
| <a href="https://zenodo.org/record/7262581">AMOS</a>                                                                                         |       600 | Abdomen                             | CT, MRI                         |
| <a href="https://ieee-dataport.org/open-access/bimcv-covid-19-large-annotated-dataset-rx-and-ct-images-covid-19-patients-0">BIMCVCovid19</a> |     16335 | Chest                               | CT                              |
| <a href="https://sites.google.com/view/calgary-campinas-dataset/home">CC359</a>                                                              |       359 | Head                                | MRI T1                          |
| <a href="https://physionet.org/content/ct-ich/1.3.1/">CT_ICH</a>                                                                             |        75 | Head                                | CT                              |
| <a href="https://zenodo.org/record/6504722#.YsgwnNJByV4">CrossMoDA</a>                                                                       |       484 | Head                                | MRI T1c, MRI T2hr               |
| <a href="https://xnat.bmia.nl/data/archive/projects/egd">EGD</a>                                                                             |       774 | Head                                | FLAIR, MRI T1, MRI T1GD, MRI T2 |
| <a href="https://flare22.grand-challenge.org/">FLARE2022</a>                                                                                 |      2100 | Abdomen                             | CT                              |
| <a href="https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=1966254">LIDC</a>                                                |      1018 | Chest                               | CT                              |
| <a href="https://competitions.codalab.org/competitions/17094">LiTS</a>                                                                       |       201 | Abdominal                           | CT                              |
| <a href="https://www.medseg.ai/database/liver-segments-50-cases">LiverMedseg</a>                                                             |        50 | Chest, Abdomen                      | CT                              |
| <a href="https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=80969742">MIDRC</a>                                              |       155 | Thorax                              | CT                              |
| <a href="http://medicalood.dkfz.de/web/">MOOD</a>                                                                                            |      1358 | Head, Abdominal                     | MRI, CT                         |
| <a href="http://medicalsegmentation.com/covid19/">Medseg9</a>                                                                                |         9 | Chest                               | CT                              |
| <a href="https://mosmed.ai/en/datasets/ct_lungcancer_500/">MoscowCancer500</a>                                                               |       979 | Thorax                              | CT                              |
| <a href="https://mosmed.ai/en/datasets/covid191110/">MoscowCovid1110</a>                                                                     |      1110 | Thorax                              | CT                              |
| <a href="https://wiki.cancerimagingarchive.net/display/NLST/National+Lung+Screening+Trial">NLST</a>                                          |     13623 | Thorax                              | CT                              |
| <a href="https://wiki.cancerimagingarchive.net/display/Public/NSCLC-Radiomics">NSCLC</a>                                                     |       422 | Thorax                              | CT                              |
| <a href="https://www.kaggle.com/competitions/rsna-breast-cancer-detection/data">RSNABreastCancer</a>                                         |     54710 | Thorax                              | MG                              |
| <a href="https://zenodo.org/record/6802614#.Y6M2MxXP1D8">Totalsegmentator</a>                                                                |      1204 | Head, Thorax, Abdomen, Pelvis, Legs | CT                              |
| <a href="https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70229053">VSSEG</a>                                              |       242 | Head                                | MRI T1c, MRI T2                 |
| <a href="https://osf.io/4skx2/">VerSe</a>                                                                                                    |       374 | Thorax, Abdomen                     | CT                              |

Check out [our docs](https://neuro-ml.github.io/amid/) for a more detailed list of available datasets and their fields.

# Install

Just get it from PyPi:

```shell
pip install amid
```

Or if you want to use version control features:

```shell
git clone https://github.com/neuro-ml/amid.git
cd amid && pip install -e .
```

# Contribute

Check our [contribution guide](https://neuro-ml.github.io/amid/latest/CONTRIBUTING/) if you want to add a new dataset to
AMID.
