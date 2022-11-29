import gzip
import json
import tarfile
import typing as tp
from functools import lru_cache
from pathlib import Path

import nibabel as nb
import numpy as np
import pandas as pd
import pydicom
from connectome import Source, meta, Transform
from deli import load

from .internals import checksum, register


# @register(
#     body_region='Chest',
#     license='',
#     link=['https://github.com/BIMCV-CSUSP/BIMCV-COVID-19', 
#           'https://ieee-dataport.org/open-access/bimcv-covid-19-large-annotated-dataset-rx-and-ct-images-covid-19-patients-0'],
#     modality='CT',
#     prep_data_size='859G',
#     raw_data_size='859G',
#     task='Segmentation',
# )
# @checksum('bimcv_covid19')
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

    @meta
    def ids(_series2metainfo):
        return tuple(sorted(_series2metainfo))

    # @meta
    # def ids(_pos_root, _neg_root):
    #     pos_ids = load(_pos_root / 'pos_good_50_ids.json')
    #     neg_ids = load(_neg_root / 'neg_good_ids.json')
    #     return sorted(pos_ids + neg_ids)

    def session_id(key, _series2metainfo):
        return _series2metainfo[key]['session_id']

    def subject_id(key, _series2metainfo):
        return _series2metainfo[key]['subject_id']

    @lru_cache(None)
    def _series2metainfo(_base):
        """
        dict with series_id : {'session_id': ...
                                'subject_id': ...,
                                'archive_path': ...,
                                'is_positive': ...,
                                'image_path': ...,
                                'meta_path': ...}
        """
        series2metainfo = {}
        for part_filename in _base.glob('**/*part*.tar.gz'):
            if 'pos' in str(part_filename):
                is_positive = True  
            else:
                assert 'neg' in str(part_filename)
                is_positive = False


            text_descr_splits = load(str(part_filename) + '.tar-tvf.txt').split()

            for el_desrc in text_descr_splits:
                # only chest ct images ('.json' and '.nii.gz' files)
                # 'don't remove png but it should not be in cases'
                if 'chest_ct' not in el_desrc or not el_desrc.endswith(('.nii.gz', '.json')):
                    continue

                image_path = None
                meta_path = None
                if el_desrc.endswith('.nii.gz'):
                    ext = '.nii.gz'
                    image_path = el_desrc
                # elif el_desrc.endswith(".json"):
                else:
                    assert el_desrc.endswith('.json')
                    ext = '.json'
                    meta_path = el_desrc

                # obtaining series id
                series_id = str(Path(el_desrc).name)[: -len(ext)]

                # if no such series yet
                if series_id not in series2metainfo:
                    # obtain subject_id, session_id
                    # TODO: make sure subject_id and session_id are defined here
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

    # `extractfile` is the most expensive part here, so we better call it only once
    def _nii_data(key, _series2metainfo):
        with tarfile.open(_series2metainfo[key]['archive_path']) as part_file:
            data = part_file.extractfile(_series2metainfo[key]['image_path'])
            with gzip.GzipFile(fileobj=data) as nii:
                nii = nb.FileHolder(fileobj=nii)
                image = nb.Nifti1Image.from_file_map({'header': nii, 'image': nii})
                return np.asarray(image.dataobj), image.affine

    def image(_nii_data):
        return _nii_data[0]

    def affine(_nii_data):
        return _nii_data[1]

    def tags(key, _series2metainfo) -> dict:
        """
        dicom tags
        """
        try:
            with tarfile.open(_series2metainfo[key]['archive_path']) as part_file:
                data = part_file.extractfile(_series2metainfo[key]['meta_path'])
                tags_dict = json.loads(data.read().decode('utf-8'))

                return parse_dicom_tags(tags_dict)

        except ValueError:
            return {}

    @lru_cache(None)
    def _labels_dataframe(_root):

        pos_labels_tarfile_name,  = _root.glob('**/covid19_posi_head.tar.gz')
        pos_labels_tarfile_subpath = 'covid19_posi/derivatives/labels/labels_covid_posi.tsv'
        with tarfile.open(pos_labels_tarfile_name) as file:
            pos_dataframe = pd.read_csv(file.extractfile(pos_labels_tarfile_subpath),
                                        sep='\t', index_col='ReportID').iloc[:, 1:]

        neg_labels_tarfile_name,  = _root.glob('**/covid19_neg_derivative.tar.gz')
        neg_labels_tarfile_subpath = 'covid19_neg/derivatives/labels/Labels_covid_NEG_JAN21.tsv'
        with tarfile.open(neg_labels_tarfile_name) as file:
            neg_dataframe = pd.read_csv(file.extractfile(neg_labels_tarfile_subpath),
                                        sep='\t', index_col='ReportID').iloc[:, 1:]

        return pd.concat([pos_dataframe, neg_dataframe], ignore_index=False)

    def label_info(key, _series2metainfo, _labels_dataframe) -> dict:
        """
        labelCUIS, Report, LocalizationsCUIS etc.
        """
        session_id = _series2metainfo[key]['session_id']

        if session_id in _labels_dataframe.index:
            return dict(_labels_dataframe.loc[session_id])
        else:
            return {}

    @lru_cache(None)
    def _subject_df(_root):

        pos_subjects_tarfile_name,  = _root.glob('**/covid19_posi_subjects.tar.gz')
        pos_subjects_tarfile_subpath = 'covid19_posi/participants.tsv'
        with tarfile.open(pos_subjects_tarfile_name) as file:
            pos_data = pd.read_csv(file.extractfile(pos_subjects_tarfile_subpath),
                                   sep='\t', index_col='participant')
            pos_data = pos_data[pos_data.index != 'derivatives']

        neg_subjects_tarfile_name, = _root.glob('**/covid19_neg_metadata.tar.gz')
        neg_subjects_tarfile_subpath = 'covid19_neg/participants.tsv'
        with tarfile.open(neg_subjects_tarfile_name) as file:
            neg_data = pd.read_csv(file.extractfile(neg_subjects_tarfile_subpath),
                                   sep='\t', index_col='participant')
            neg_data = neg_data[neg_data.index != 'derivatives']

        data = pd.concat([pos_data, neg_data])

        return data

    def subject_info(key, _series2metainfo, _subject_df) -> dict:
        """
        modality_dicom (=[CT]), body_parts(=[chest]), age, gender
        """
        subject_id = _series2metainfo[key]['subject_id']
        if subject_id in _subject_df.index:
            return dict(_subject_df.loc[subject_id])
        else:
            return {}

    def session_info(key, _root, _series2metainfo) -> dict:
        """
        study_date,	medical_evaluation
        """
        
        session_id = _series2metainfo[key]['session_id']
        subject_id = _series2metainfo[key]['subject_id']

        pos_sessions_tarfile_name, = _root.glob('**/covid19_posi_sessions_tsv.tar.gz')
        neg_sessions_tarfile_name, = _root.glob('**/covid19_neg_sessions_tsv.tar.gz')

        if _series2metainfo[key]['is_positive']:
            step_sessions_tarfile_name = pos_sessions_tarfile_name
        else:
            step_sessions_tarfile_name = neg_sessions_tarfile_name

        with tarfile.open(step_sessions_tarfile_name) as all_sessions_file:
            # TODO: potentially you may want to load this from the associated txt file

            txt_splits = load(str(step_sessions_tarfile_name) + '.tar-tvf.txt').split()

            ses_file, = filter(lambda x: subject_id in x, txt_splits)
            sesions_dataframe = pd.read_csv(all_sessions_file.extractfile(ses_file), sep="\t", index_col='session_id')

        if session_id in sesions_dataframe.index:
            return dict(sesions_dataframe.loc[session_id])
        else:
            return {}


class SpacingFromAffine(Transform):
    __inherit__ = True

    def spacing(affine):
        return nb.affines.voxel_sizes(affine)


def parse_dicom_tags(tags: tp.Dict[str, tp.Any]) -> tp.Optional[tp.Union[dict, list]]:
    if not isinstance(tags, dict):
        return tags
    if len(tags) == 1:
        if 'vr' in tags:
            return None
        return parse_dicom_tags(list(tags.values())[0])

    if set(tags.keys()) == {'Value', 'vr'}:
        value = tags['Value']
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
