import os
import typing as tp
import gzip
import tarfile
import json
from functools import lru_cache

from pathlib import Path
import pydicom
import nibabel as nb

import numpy as np
import pandas as pd
from connectome import Source, meta

from dpipe.io import load

from amid.internals import checksum, register



def parse_dicom_tags(tags: tp.Dict[str, tp.Any]) -> tp.Optional[tp.Union[dict, list]]:
    if not isinstance(tags, dict):
        return tags
    if len(tags) == 1:
        if "vr" in tags:
            return None
        return parse_dicom_tags(list(tags.values())[0])

    if set(tags.keys()) == {"Value", "vr"}:
        value = tags["Value"]
        # vr = tags["vr"]
        assert isinstance(value, list)
        result = [parse_dicom_tags(v) for v in value]
        if not result:
            return None
        if len(result) == 1:
            return result[0]
        return result
    result_: tp.Dict[str, tp.Optional[tp.Union[dict, list]]] = {}
    for tag, value in tags.items():
        try:
            keyword = pydicom.datadict.keyword_for_tag(tag)
        except ValueError:
            keyword = str(tag)
        assert isinstance(result_, dict)
        result_[keyword] = parse_dicom_tags(value)
    return result_


@register(
    body_region='Chest',
    license='',
    link=['https://github.com/BIMCV-CSUSP/BIMCV-COVID-19', 
          'https://ieee-dataport.org/open-access/bimcv-covid-19-large-annotated-dataset-rx-and-ct-images-covid-19-patients-0'],
    modality='CT',
    prep_data_size='859G',
    raw_data_size='859G',
    task='Segmentation',
)
@checksum('bimcv_covid19')
class BIMCVCovid19(Source):
    _root: str
    """
    BIMCV COVID-19 Dataset, CT-images only
    It includes BIMCV COVID-19 positive partition (https://arxiv.org/pdf/2006.01174.pdf)
    and negative partion (https://ieee-dataport.org/open-access/bimcv-covid-19-large-annotated-dataset-rx-and-ct-images-covid-19-patients-0)

    PCR tests are not used
    
    Parameters
    ----------
    root : str, Path
        path to the folder containing the downloaded and parsed data.
        
    Notes
    -----
    Dataset has 2 partitions: bimcv-covid19-positive and bimcv-covid19-positive
    Each partition is spread over the 81 different tgz archives. The archives includes metadata about
    subject, sessions, and labels. Also there are some tgz archives for nifty images in nii.gz format
    
    Examples
    --------
    >>> # Place the downloaded archives in any folder and pass the path to the constructor:
    >>> ds = BIMCVCovid19(root='/path/to/downloaded/data/folder/')
    >>> print(len(ds.ids))
    # 201
    >>> print(ds.image(ds.ids[0]).shape)
    # (512, 512, 163)
    >>> print(ds.is_positive(ds.ids[0]))
    # True
    >>> print(ds.subject_info[80])
    # {'modality_dicom': "['CT']",
    #  'body_parts': "[['chest']]",
    #  'age': '[80]',
    #  'gender': 'M'}

    References
    ----------
    .. [1] Maria De La Iglesia Vayá, Jose Manuel Saborit, Joaquim Angel Montell, Antonio Pertusa, Aurelia
            Bustos, Miguel Cazorla, Joaquin Galant, Xavier Barber, Domingo Orozco-Beltrán, Francisco
            Garcia, Marisa Caparrós, Germán González, and Jose María Salinas. BIMCV COVID-19+: a
            large annotated dataset of RX and CT images from COVID-19 patients. arXiv:2006.01174, 2020.
    .. [2] Maria de la Iglesia Vayá, Jose Manuel Saborit-Torres, Joaquim Angel Montell Serrano, 
            Elena Oliver-Garcia, Antonio Pertusa, Aurelia Bustos, Miguel Cazorla, Joaquin Galant,
            Xavier Barber, Domingo Orozco-Beltrán, Francisco García-García, Marisa Caparrós, Germán González,
            Jose María Salinas, 2021. BIMCV COVID-19-: a large annotated dataset of RX and CT images from COVID-19 patients.
            Available at: https://dx.doi.org/10.21227/m4j2-ap59.
    """
    def _base(_root):
        return Path(_root)
    
    def _pos_root(_base):
        return _base / 'bimcv-covid19-positive' / 'original'
    
    def _neg_root(_base):
        return _base / 'bimcv-covid19-negative' / 'original'
    
    def _pos_subjects_tarfile_name(_pos_root):
        return _pos_root / "covid19_posi_subjects.tar.gz"

    def _pos_subjects_tarfile_subpath():
        return "covid19_posi/participants.tsv"
    
    def _pos_sessions_tarfile_name(_pos_root):
        return _pos_root / "covid19_posi_sessions_tsv.tar.gz"
    
    def _pos_tests_tarfile_name(_pos_root):
        return _pos_root / "covid19_posi_head.tar.gz"

    def _tests_tarfile_subpath():
        return "covid19_posi/derivatives/EHR/sil_reg_covid_posi.tsv"
    
    def _pos_labels_tarfile_name(_pos_root):
        return _pos_root / "covid19_posi_head.tar.gz"
    
    def _pos_labels_tarfile_subpath():
        return "covid19_posi/derivatives/labels/labels_covid_posi.tsv"
    
    def _neg_labels_tarfile_name(_neg_root):
        return _neg_root / "covid19_neg_derivative.tar.gz"
    
    def _neg_labels_tarfile_subpath():
        return "covid19_neg/derivatives/labels/Labels_covid_NEG_JAN21.tsv"
    
    def _neg_subjects_tarfile_name(_neg_root):
        return _neg_root / "covid19_neg_metadata.tar.gz"
        
    def _neg_subjects_tarfile_subpath():
        return "covid19_neg/participants.tsv"
    
    def _neg_sessions_tarfile_name(_neg_root):
        return _neg_root / "covid19_neg_sessions_tsv.tar.gz"
    
#     @meta
#     def ids(_pos_root, _neg_root):
#         ids = []

#         part_filenames = sorted(list(_pos_root.glob("*part*.tar.gz")) +\
#                                 list(_neg_root.glob("*part*.tar.gz")))

#         for part_filename in part_filenames:
#             with tarfile.open(part_filename) as part_file:
#                 members = part_file.getmembers()
                
#                 for member in members:
#                     if not member.isfile():
#                         continue

#                     member_path = member.name

#                     #only chest ct images ('.json' and '.nii.gz' files) 'don't remove png but it should not be in cases'
#                     if not member_path.endswith(".nii.gz") or 'chest_ct' not in member_path:
#                         continue

#                     ids.append(Path(member_path).name[: -len(".nii.gz")])
                    
#         return ids
    @meta
    def ids(_pos_root, _neg_root):
        pos_ids = load(_pos_root / 'pos_good_50_ids.json')
        neg_ids = load(_neg_root / 'neg_good_ids.json')
        return sorted(pos_ids + neg_ids)
    
    def session_id(key, _series2metainfo):
        return _series2metainfo[key]['session_id']
    
    def subject_id(key, _series2metainfo):
        return _series2metainfo[key]['subject_id']
    
    @lru_cache(None)
    def _series2metainfo(_pos_root, _neg_root):
        """
        dict with series_id : {'session_id': ...
                                'subject_id': ...,
                                'archive_path': ...,
                                'is_positive': ...,
                                'image_path': ...,
                                'meta_path': ...}
        """
        series2metainfo = {}

        for is_positive, iter_root in zip([True, False],
                                    [_pos_root, _neg_root]):
            iter_part_filenames = list(iter_root.glob("*part*.tar.gz"))
            for part_filename in iter_part_filenames:
                with tarfile.open(part_filename) as part_file:
                    members = part_file.getmembers()

                    for member in members:
                        if not member.isfile():
                            continue

                        member_path = member.name

                        if 'chest_ct' not in member_path or not (member_path.endswith(".nii.gz") or member_path.endswith(".json")):
                            continue

                        image_path = None
                        meta_path = None
                        if member_path.endswith(".nii.gz"):
                            ext = ".nii.gz"
                            image_path = member_path
                        elif member_path.endswith(".json"):
                            ext = ".json"
                            meta_path = member_path
                        else:
                            raise Exception("not expected suffix: not nii.gz or json")
                        
                        #obtaining series id
                        series_id = str(Path(member_path).name)[: -len(ext)]
                        
                        # if no such series yet
                        if series_id not in series2metainfo:

                            #obtain subject_id, session_id
                            for el in series_id.split('_'):
                                if 'sub' in el:
                                    subject_id = el
                                elif 'ses' in el:
                                    session_id = el

                            series2metainfo[series_id] = {
                                'session_id': session_id,
                                'subject_id': subject_id,
                                'archive_path': part_filename,
                                'is_positive': is_positive,
                                'image_path': None,
                                'meta_path': None,
                            }

                        if image_path is not None:
                            series2metainfo[series_id]['image_path'] = image_path

                        if meta_path is not None:
                            series2metainfo[series_id]['meta_path'] = meta_path
        return series2metainfo
    
    def is_positive(key, _series2metainfo):

        return _series2metainfo[key]['is_positive']
    
    def image(key, _series2metainfo):
        with tarfile.open(_series2metainfo[key]['archive_path']) as part_file:
            data = part_file.extractfile(_series2metainfo[key]['image_path'])
            with gzip.GzipFile(fileobj=data) as nii:
                nii = nb.FileHolder(fileobj=nii)
                image = nb.Nifti1Image.from_file_map({'header': nii, 'image': nii})

                return np.int16(image.get_fdata())
    
    def affine(key, _series2metainfo):
        with tarfile.open(_series2metainfo[key]['archive_path']) as part_file:
            data = part_file.extractfile(_series2metainfo[key]['image_path'])
            with gzip.GzipFile(fileobj=data) as nii:
                nii = nb.FileHolder(fileobj=nii)
                image = nb.Nifti1Image.from_file_map({'header': nii, 'image': nii})

                return image.affine
            
    def spacing(key, _series2metainfo):
        with tarfile.open(_series2metainfo[key]['archive_path']) as part_file:
            data = part_file.extractfile(_series2metainfo[key]['image_path'])
            with gzip.GzipFile(fileobj=data) as nii:
                nii = nb.FileHolder(fileobj=nii)
                image = nb.Nifti1Image.from_file_map({'header': nii, 'image': nii})

                return image.header.get_zooms()
    
    def tags(key, _series2metainfo):
        """
        dicom tags
        """
        try:
            with tarfile.open(_series2metainfo[key]['archive_path']) as part_file:
                data = part_file.extractfile(_series2metainfo[key]['meta_path'])
                tags_dict = json.loads(data.read().decode('utf-8'))

                return parse_dicom_tags(tags_dict)

        except Exception:
            return None
        
        
    @lru_cache(None)
    def _labels_dataframe(_pos_labels_tarfile_name, _pos_labels_tarfile_subpath,
                          _neg_labels_tarfile_name, _neg_labels_tarfile_subpath):
    
        with tarfile.open(_pos_labels_tarfile_name) as file:
            pos_dataframe = pd.read_csv(file.extractfile(_pos_labels_tarfile_subpath), 
                                        sep='\t', index_col='ReportID').iloc[:, 1:]

        with tarfile.open(_neg_labels_tarfile_name) as file:
            neg_dataframe = pd.read_csv(file.extractfile(_neg_labels_tarfile_subpath), 
                                        sep='\t', index_col='ReportID').iloc[:, 1:]

        return pd.concat([pos_dataframe, neg_dataframe], ignore_index=False)

    def label_info(key, _series2metainfo, _labels_dataframe):
        """
        labelCUIS, Report, LocalizationsCUIS etc.
        """
        session_id = _series2metainfo[key]['session_id']
        
        if session_id in _labels_dataframe.index:
            return dict(_labels_dataframe.loc[session_id])
        else:
            return None
    
    @lru_cache(None)
    def _subject_df(_pos_subjects_tarfile_name, _pos_subjects_tarfile_subpath,
                    _neg_subjects_tarfile_name, _neg_subjects_tarfile_subpath):
        with tarfile.open(_pos_subjects_tarfile_name) as file:
            pos_data = pd.read_csv(file.extractfile(_pos_subjects_tarfile_subpath), 
                                   sep='\t', index_col='participant')
            pos_data = pos_data[pos_data.index != 'derivatives']

        with tarfile.open(_neg_subjects_tarfile_name) as file:
            neg_data = pd.read_csv(file.extractfile(_neg_subjects_tarfile_subpath),
                                   sep='\t', index_col='participant')
            neg_data = neg_data[neg_data.index != 'derivatives']

        data = pd.concat([pos_data, neg_data])

        return data

    def subject_info(key, _series2metainfo, _subject_df):
        """
        modality_dicom (=[CT]), body_parts(=[chest]), age, gender
        """
        subject_id = _series2metainfo[key]['subject_id']
        if subject_id in _subject_df.index:
            return dict(_subject_df.loc[subject_id])
        else:
            return None
    
    def session_info(key, _series2metainfo, 
                       _pos_sessions_tarfile_name,
                      _neg_sessions_tarfile_name):
        """
        study_date,	medical_evaluation
        """
        session_id = _series2metainfo[key]['session_id']
        subject_id = _series2metainfo[key]['subject_id']

        if _series2metainfo[key]['is_positive']:
            step_sessions_tarfile_name = _pos_sessions_tarfile_name
        else:
            step_sessions_tarfile_name = _neg_sessions_tarfile_name

        with tarfile.open(step_sessions_tarfile_name) as all_sessions_file:
            ses_members = all_sessions_file.getmembers()

            for ses_file in ses_members:
                if subject_id in ses_file.name:
                    break
            sesions_dataframe = pd.read_csv(all_sessions_file.extractfile(ses_file), sep="\t", index_col='session_id')

        if session_id in sesions_dataframe.index:
            return dict(sesions_dataframe.loc[session_id])
        else:
            return None

    def cancer_mask(key):
        return None
    
    def covid_mask(key):
        return None
