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


ds = VerSe(root='/path/to/raw/data')
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

| Name                                                                                                                               |   Entries | Body region                         | Modality                                                               |
|:-----------------------------------------------------------------------------------------------------------------------------------|----------:|:------------------------------------|:-----------------------------------------------------------------------|
| <a href="https://neuro-ml.github.io/amid/latest/datasets-api/#amid.amos.dataset.AMOS">AMOS</a>                                     |      2465 | Abdomen                             | CT, MRI                                                                |
| <a href="https://neuro-ml.github.io/amid/latest/datasets-api/#amid.bimcv.BIMCVCovid19">BIMCVCovid19</a>                            |     16364 | Chest                               | CT                                                                     |
| <a href="https://neuro-ml.github.io/amid/latest/datasets-api/#amid.brats2021.BraTS2021">BraTS2021</a>                              |      5880 | Head                                | MRI T1, MRI T1Gd, MRI T2, MRI T2-FLAIR                                 |
| <a href="https://neuro-ml.github.io/amid/latest/datasets-api/#amid.cc359.dataset.CC359">CC359</a>                                  |       359 | Head                                | MRI T1                                                                 |
| <a href="https://neuro-ml.github.io/amid/latest/datasets-api/#amid.cl_detection.CLDetection2023">CLDetection2023</a>               |       400 | Head                                | X-ray                                                                  |
| <a href="https://neuro-ml.github.io/amid/latest/datasets-api/#amid.crlm.CRLM">CRLM</a>                                             |       197 | Abdomen                             | CT, SEG                                                                |
| <a href="https://neuro-ml.github.io/amid/latest/datasets-api/#amid.ct_ich.CT_ICH">CT_ICH</a>                                       |        75 | Head                                | CT                                                                     |
| <a href="https://neuro-ml.github.io/amid/latest/datasets-api/#amid.curvas.CURVAS">CURVAS</a>                                       |        90 | Abdomen                             | CT                                                                     |
| <a href="https://neuro-ml.github.io/amid/latest/datasets-api/#amid.crossmoda.CrossMoDA">CrossMoDA</a>                              |       484 | Head                                | MRI T1c, MRI T2hr                                                      |
| <a href="https://neuro-ml.github.io/amid/latest/datasets-api/#amid.deeplesion.DeepLesion">DeepLesion</a>                           |     20094 | Abdomen, Thorax                     | CT                                                                     |
| <a href="https://neuro-ml.github.io/amid/latest/datasets-api/#amid.egd.EGD">EGD</a>                                                |      3096 | Head                                | FLAIR, MRI T1, MRI T1GD, MRI T2                                        |
| <a href="https://neuro-ml.github.io/amid/latest/datasets-api/#amid.flare2022.FLARE2022">FLARE2022</a>                              |      2100 | Abdomen                             | CT                                                                     |
| <a href="https://neuro-ml.github.io/amid/latest/datasets-api/#amid.hcp.HCP">HCP</a>                                                |      1113 | Head                                | MRI                                                                    |
| <a href="https://neuro-ml.github.io/amid/latest/datasets-api/#amid.kits.KiTS23">KiTS23</a>                                         |       489 | thorax                              | CT                                                                     |
| <a href="https://neuro-ml.github.io/amid/latest/datasets-api/#amid.lidc.dataset.LIDC">LIDC</a>                                     |      1018 | Chest                               | CT                                                                     |
| <a href="https://neuro-ml.github.io/amid/latest/datasets-api/#amid.lits.dataset.LiTS">LiTS</a>                                     |       201 | Abdominal                           | CT                                                                     |
| <a href="https://neuro-ml.github.io/amid/latest/datasets-api/#amid.liver_medseg.LiverMedseg">LiverMedseg</a>                       |        50 | Chest, Abdomen                      | CT                                                                     |
| <a href="https://neuro-ml.github.io/amid/latest/datasets-api/#amid.midrc.MIDRC">MIDRC</a>                                          |       229 | Thorax                              | CT                                                                     |
| <a href="https://neuro-ml.github.io/amid/latest/datasets-api/#amid.mood.MOOD">MOOD</a>                                             |      1358 | Head, Abdominal                     | MRI, CT                                                                |
| <a href="https://neuro-ml.github.io/amid/latest/datasets-api/#amid.msd.MSD">MSD</a>                                                |      2628 | Chest, Abdominal, Head              | CT, CE CT, MRI, MRI FLAIR, MRI T1w, MRI t1gd, MRI T2w, MRI T2, MRI ADC |
| <a href="https://neuro-ml.github.io/amid/latest/datasets-api/#amid.mslub.dataset.MSLUB">MSLUB</a>                                  |        70 | Head                                | MRI                                                                    |
| <a href="https://neuro-ml.github.io/amid/latest/datasets-api/#amid.medseg9.Medseg9">Medseg9</a>                                    |         9 | Chest                               | CT                                                                     |
| <a href="https://neuro-ml.github.io/amid/latest/datasets-api/#amid.cancer_500.dataset.MoscowCancer500">MoscowCancer500</a>         |       979 | Thorax                              | CT                                                                     |
| <a href="https://neuro-ml.github.io/amid/latest/datasets-api/#amid.covid_1110.MoscowCovid1110">MoscowCovid1110</a>                 |      1110 | Thorax                              | CT                                                                     |
| <a href="https://neuro-ml.github.io/amid/latest/datasets-api/#amid.nlst.NLST">NLST</a>                                             |     26254 | Thorax                              | CT                                                                     |
| <a href="https://neuro-ml.github.io/amid/latest/datasets-api/#amid.nsclc.NSCLC">NSCLC</a>                                          |       422 | Thorax                              | CT                                                                     |
| <a href="https://neuro-ml.github.io/amid/latest/datasets-api/#amid.rsna_bc.dataset.RSNABreastCancer">RSNABreastCancer</a>          |     54710 | Thorax                              | MG                                                                     |
| <a href="https://neuro-ml.github.io/amid/latest/datasets-api/#amid.ribfrac.dataset.RibFrac">RibFrac</a>                            |       660 | Chest                               | CT                                                                     |
| <a href="https://neuro-ml.github.io/amid/latest/datasets-api/#amid.stanford_coca.StanfordCoCa">StanfordCoCa</a>                    |      1000 | Coronary, Chest                     | CT                                                                     |
| <a href="https://neuro-ml.github.io/amid/latest/datasets-api/#amid.tbad.TBAD">TBAD</a>                                             |       100 | Chest                               | CT                                                                     |
| <a href="https://neuro-ml.github.io/amid/latest/datasets-api/#amid.totalsegmentator.dataset.Totalsegmentator">Totalsegmentator</a> |      1204 | Head, Thorax, Abdomen, Pelvis, Legs | CT                                                                     |
| <a href="https://neuro-ml.github.io/amid/latest/datasets-api/#amid.upenn_gbm.upenn_gbm.UPENN_GBM">UPENN_GBM</a>                    |       671 | Head                                | FLAIR, MRI T1, MRI T1GD, MRI T2, DSC MRI, DTI MRI                      |
| <a href="https://neuro-ml.github.io/amid/latest/datasets-api/#amid.vs_seg.dataset.VSSEG">VSSEG</a>                                 |       484 | Head                                | MRI T1c, MRI T2                                                        |
| <a href="https://neuro-ml.github.io/amid/latest/datasets-api/#amid.verse.VerSe">VerSe</a>                                          |       374 | Thorax, Abdomen                     | CT                                                                     |

Check out [our docs](https://neuro-ml.github.io/amid/) for a more detailed list of available datasets and their fields.

# Install

Just get it from PyPi:

```shell
pip install amid
```

# Contribute

Check our [contribution guide](https://neuro-ml.github.io/amid/latest/CONTRIBUTING/) if you want to add a new dataset to
AMID.
