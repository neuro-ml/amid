import contextlib
import gzip
import json
import tarfile
import typing as tp
from functools import cached_property
from pathlib import Path

import nibabel as nb
import numpy as np
import pandas as pd
import pydicom
from deli import load

from .internals import Dataset, field, licenses, register


@register(
    body_region='Chest',
    license=licenses.CC_BY_40,
    link='https://ieee-dataport.org/open-access/bimcv-covid-19'
    '-large-annotated-dataset-rx-and-ct-images-covid-19-patients-0',
    modality='CT',
    prep_data_size='859G',
    raw_data_size='859G',
    task='Segmentation',
)
class BIMCVCovid19(Dataset):
    """
    BIMCV COVID-19 Dataset, CT-images only
    It includes BIMCV COVID-19 positive partition (https://arxiv.org/pdf/2006.01174.pdf)
    and negative partion
    (https://ieee-dataport.org/open-access/bimcv-covid-19-large-annotated-dataset-rx-and-ct-images-covid-19-patients-0)

    PCR tests are not used

    GitHub page: https://github.com/BIMCV-CSUSP/BIMCV-COVID-19

    Parameters
    ----------
    root : str, Path
        path to the folder containing the downloaded and parsed data.

    Notes
    -----
    Dataset has 2 partitions: bimcv-covid19-positive and bimcv-covid19-positive
    Each partition is spread over the 81 different tgz archives. The archives include metadata about
    subject, sessions, and labels. Also, there are some tgz archives for nifty images in nii.gz format

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
            Jose María Salinas, 2021. BIMCV COVID-19-: a large annotated dataset of RX and CT images from COVID-19
            patients.
            Available at: https://dx.doi.org/10.21227/m4j2-ap59.
    """

    @property
    def ids(self):
        return tuple(sorted(self._series2metainfo))

    @field
    def session_id(self, i):
        return self._series2metainfo[i]['session_id']

    @field
    def subject_id(self, i):
        return self._series2metainfo[i]['subject_id']

    @field
    def is_positive(self, i):
        return self._series2metainfo[i]['is_positive']

    @field
    def image(self, i):
        meta = self._series2metainfo[i]
        with unpack(self._current_root(i), meta['archive_path'], meta['image_path']) as (file, unpacked):
            if unpacked:
                array = np.asarray(nb.load(file).dataobj)
            else:
                with gzip.GzipFile(fileobj=file) as nii:
                    nii = nb.FileHolder(fileobj=nii)
                    image = nb.Nifti1Image.from_file_map({'header': nii, 'image': nii})
                    array = np.asarray(image.dataobj)

            integer = np.int16(array)
            if (array == integer).all():
                return integer

            # there are about 30 images that will visit this branch, so it's ok
            integer = np.int32(array)
            assert (array == integer).all(), np.abs(array - integer.astype(float)).max()
            return integer

    @field
    def affine(self, i):
        meta = self._series2metainfo[i]
        with unpack(self._current_root(i), meta['archive_path'], meta['image_path']) as (file, unpacked):
            if unpacked:
                return nb.load(file).affine

            with gzip.GzipFile(fileobj=file) as nii:
                nii = nb.FileHolder(fileobj=nii)
                image = nb.Nifti1Image.from_file_map({'header': nii, 'image': nii})
                return image.affine

    @field
    def tags(self, i) -> dict:
        """
        dicom tags
        """
        meta = self._series2metainfo[i]
        if meta['meta_path'] is None:
            return {}

        with unpack(self._current_root(i), meta['archive_path'], meta['meta_path']) as (file, _):
            try:
                return parse_dicom_tags(load(file))
            except ValueError:
                return {}

    @cached_property
    def _labels_dataframe(self):
        with unpack(
            self._positive_root, 'covid19_posi_head.tar.gz', 'covid19_posi/derivatives/labels/labels_covid_posi.tsv'
        ) as (file, _):
            pos_dataframe = pd.read_csv(file, sep='\t', index_col='ReportID').iloc[:, 1:]

        with unpack(
            self._negative_root,
            'covid19_neg_derivative.tar.gz',
            'covid19_neg/derivatives/labels/Labels_covid_NEG_JAN21.tsv',
        ) as (file, _):
            neg_dataframe = pd.read_csv(file, sep='\t', index_col='ReportID').iloc[:, 1:]

        return pd.concat([pos_dataframe, neg_dataframe], ignore_index=False)

    @field
    def label_info(self, i) -> dict:
        """
        labelCUIS, Report, LocalizationsCUIS etc.
        """
        session_id = self._series2metainfo[i]['session_id']

        if session_id in self._labels_dataframe.index:
            return dict(self._labels_dataframe.loc[session_id])
        else:
            return {}

    @cached_property
    def _subject_df(self):
        with unpack(self._positive_root, 'covid19_posi_subjects.tar.gz', 'covid19_posi/participants.tsv') as (file, _):
            pos_data = pd.read_csv(file, sep='\t', index_col='participant')
            pos_data = pos_data[pos_data.index != 'derivatives']

        with unpack(self._negative_root, 'covid19_neg_metadata.tar.gz', 'covid19_neg/participants.tsv') as (file, _):
            neg_data = pd.read_csv(file, sep='\t', index_col='participant')
            neg_data = neg_data[neg_data.index != 'derivatives']

        return pd.concat([pos_data, neg_data])

    @field
    def subject_info(self, i) -> dict:
        """
        modality_dicom (=[CT]), body_parts(=[chest]), age, gender
        """
        subject_id = self._series2metainfo[i]['subject_id']
        if subject_id in self._subject_df.index:
            return dict(self._subject_df.loc[subject_id])
        else:
            return {}

    @field
    def age(self, i) -> int:
        """Minimum of (possibly two) available ages.
        The maximum difference between max and min age for every patient is 1 year."""
        return min(json.loads(self.subject_info(i).get('age')))

    @field
    def sex(self, i) -> str:
        return self.subject_info(i).get('gender')

    @field
    def session_info(self, i) -> dict:
        """
        study_date,	medical_evaluation
        """
        meta = self._series2metainfo[i]
        current_root = self._current_root(i)
        session_id = meta['session_id']
        subject_id = meta['subject_id']

        if meta['is_positive']:
            step_sessions_tarfile_name = 'covid19_posi_sessions_tsv.tar.gz'
        else:
            step_sessions_tarfile_name = 'covid19_neg_sessions_tsv.tar.gz'

        txt_splits = load(current_root / (step_sessions_tarfile_name + '.tar-tvf.txt')).split()
        (ses_file,) = filter(lambda x: subject_id in x, txt_splits)

        with unpack(current_root, step_sessions_tarfile_name, ses_file) as (file, _):
            sesions_dataframe = pd.read_csv(file, sep='\t', index_col='session_id')

        if session_id in sesions_dataframe.index:
            return dict(sesions_dataframe.loc[session_id])
        return {}

    @cached_property
    def _series2metainfo(self):
        """
        Main function that gathers the metadata for the whole dataset.

        Returns
        -------
        A dict with {series_id : {'session_id': ...
                                'subject_id': ...,
                                'archive_path': ...,
                                'is_positive': ...,
                                'image_path': ...,
                                'meta_path': ...}}
        """
        series2metainfo = {}
        for part in self._positive_root, self._negative_root:
            for structure in part.glob('*part*.tar.gz.tar-tvf.txt'):
                part_filename = structure.name[: -len('.tar-tvf.txt')]
                if 'pos' in part_filename:
                    is_positive = True
                else:
                    assert 'neg' in part_filename
                    is_positive = False

                text_descr_splits = load(structure).split()
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
                        (subject_id,) = filter(lambda x: 'sub' in x, series_id.split('_'))
                        (session_id,) = filter(lambda x: 'ses' in x, series_id.split('_'))

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

    @cached_property
    def _positive_root(self):
        return find_subroot(self.root, 'covid19_posi')

    @cached_property
    def _negative_root(self):
        return find_subroot(self.root, 'covid19_neg')

    def _current_root(self, i):
        return self._positive_root if self._series2metainfo[i]['is_positive'] else self._negative_root


@contextlib.contextmanager
def unpack(root: Path, archive: str, relative: str):
    unpacked = root / relative
    if unpacked.exists():
        yield unpacked, True
    else:
        # we use a buffer of 128mb to speed everything up
        with tarfile.open(root / archive, bufsize=128 * 1024**2) as part_file:
            yield part_file.extractfile(relative), False


def find_subroot(path: Path, name: str):
    """Finds a subfolder but in a bfs manner instead of dfs"""
    folders = [path]

    while folders:
        current = folders.pop(0)

        for child in current.iterdir():
            if child.name.startswith(name) and child.name.endswith('.tar-tvf.txt'):
                return current

            if child.is_dir():
                folders.append(child)

    raise FileNotFoundError(
        'No "*.tar-tvf.txt" files have been found. They are needed to gather the datasets structure.'
    )


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
