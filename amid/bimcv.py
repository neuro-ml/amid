import os
import gzip
import json
import warnings
from pathlib import Path

import numpy as np
from connectome import Source, meta
from connectome.interface.nodes import Silent

from dpipe.io import load

from .internals import checksum, register


@register(
    body_region='Chest',
    licence='',
    link=['https://github.com/BIMCV-CSUSP/BIMCV-COVID-19', 
          'https://ieee-dataport.org/open-access/bimcv-covid-19-large-annotated-dataset-rx-and-ct-images-covid-19-patients-0'],
    modality='CT',
    prep_data_size='869G',
    raw_data_size='888G',
    task='Segmentation',
)
@checksum('bimcv')
class BIMCV(Source):
    """
    BIMCV COVID-19 Dataset, CT-images only
    can be BIMCV COVID-19 positive partition (https://arxiv.org/pdf/2006.01174.pdf)
    or negative partion (https://ieee-dataport.org/open-access/bimcv-covid-19-large-annotated-dataset-rx-and-ct-images-covid-19-patients-0)
    
    downloaded and parsed data is required using https://github.com/rilshok/BIMCV-COVID-19-interface
    
    locations CUIs are not used.
    
    Parameters
    ----------
    root : str, Path
        path to the folder containing the downloaded and parsed data.
    is_positive : bool
        if it's True, than positive COVID-19 partition is used.
        if it's False than negative one is used.
        
    Notes
    -----
    Dataset has 2 partitions: bimcv-covid19-positive and bimcv-covid19-positive
    Data Format:
        <...>/{partition_name}/prepared/series/sub-S04564_ses-E09030_run-1_bp-chest_ct/image.npy.gz
        S04564 - subject_id,
        E09030 - session_id;
        information about labels: <...>/prepared/sessions/E09030/labels.json
    Examples
    --------
    >>> # Place the downloaded archives in any folder and pass the path to the constructor:
    >>> ds = LiTS(root='/path/to/downloaded/data/folder/', is_positive=True)
    >>> print(len(ds.ids))
    # 201
    >>> print(ds.image(ds.ids[0]).shape)
    # (512, 512, 163)
    >>> print(ds.labels(ds.ids[80]))
    # ['unchanged', 'cyst', 'normal', 'vertebral degenerative changes']
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
    
    _root: str
        
    def _base(_root):      
        return Path(_root)
        
    def _pos_root(_base):
        return _base / 'bimcv-covid19-positive' / 'prepared'
    
    def _neg_root(_base):
        return _base / 'bimcv-covid19-negative' / 'prepared'

    def _partition_root(key, _pos_root, _neg_root):
        if key in os.listdir(_pos_root / 'series'):
            return _pos_root
        elif key in os.listdir(_neg_root / 'series'):
            return _neg_root
        else:
            raise Exception("wrong series")

    def is_positive(key, _pos_root, _neg_root):
        if key in os.listdir(_pos_root / 'series'):
            return True
        elif key in os.listdir(_neg_root / 'series'):
            return False
        else:
            raise Exception("wrong series")
    
    def session_id(key, _partition_root):
        return load(_partition_root / 'series' / key / 'session_id.json')
    
    def subject_id(key, _partition_root):
        return load(_partition_root / 'series' / key / 'subject_id.json')
     
    def _label_info(key, _partition_root):
        session_id = load(_partition_root / 'series' / key / 'session_id.json')
        
        return load(_partition_root / 'sessions' / session_id / 'labels.json')
        
    @meta
    def ids(_pos_root, _neg_root):
        all_series = os.listdir(_pos_root / 'series') + \
                     os.listdir(_neg_root / 'series')
        
        chest_series = list(filter(lambda s: 'chest_ct' in s.lower(), all_series))
        
        return sorted(chest_series)

    def report(_label_info):
        """returns report (in Spanish)"""
        return _label_info['report']
    
    def labels(_label_info):
        """
        returns list of labels;
        Labels were automatically extracted from report using a bidirectional LSTM multi-label classifier.
        The labels correspond to biomedical vocabulary unique identifier (CUIs) codes.
        """
        return _label_info['labels']
    
    def label_CUIS(_label_info):
        """
        returns list of CUI codes corresponding to labels
        """
        return _label_info['label_CUIS']

    def tags(key, _partition_root):
        try:
            with gzip.open(_partition_root / 'series' / key / 'tags.json.gz') as f:
                return json.loads(f.read().decode('utf-8'))
        except Exception:
            return None
        
    def image(key, _partition_root):
        image = load(_partition_root / 'series' / key / 'image.npy.gz')
        return image.astype(np.int16)

    def affine(key, _partition_root):
        return load(_partition_root / 'series' / key / 'affine.npy.gz')
    
    def spacing(key, _partition_root):
        """ Returns voxel spacing along axes (x, y, z). """
        return tuple(load(_partition_root / 'series' / key / 'spacing.json'))
    
    def cancer_mask(key):
        return None
    
    def covid_mask(key):
        return None
