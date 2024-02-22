from pathlib import Path

import gzip
import numpy as np
import nibabel as nb
from connectome import Source, meta
from connectome.interface.nodes import Silent
import tarfile

from .internals import licenses, normalize


class ATLASBase(Source):
    _root: str = None

    @meta
    def ids(_root: Silent):
        archive = _root / 'ATLAS_R2.0.tar.gz'

        with tarfile.open(archive, "r:gz") as tar:
            inds = []
            path2ind = {}
            for member in tar.getmembers():
                filename = member.name
                name = filename.split('/')[-1]

                if 'T1w' in filename:
                    for stage in ['Training', 'Testing']:
                        if stage in filename:
                            path2ind[filename] = f"{stage}-{name.split('T1w')[0][:-1]}"
        return list(path2ind.values())

    def _file(i, _root: Silent):
        patient, session, filenmae = i.split('_')

        stage = patient.split('-')[0]

        code = patient.split('-')[2]
        number = code.split('s')[0].upper()

        sub = f"{patient.split('-')[1]}-{code}"
        
        path = Path('ATLAS_2') / stage / number / sub / session / 'anat' / i[(len(stage) + 1):]
        return path

    def image(_file, _root: Silent):
        archive = _root / 'ATLAS_R2.0.tar.gz'
        filename = str(_file) + '_T1w.nii.gz'
        with tarfile.open(archive, "r:gz") as tar:
            nii_gz_data = tar.extractfile(filename)
                    
            with gzip.GzipFile(fileobj=nii_gz_data) as nii:
                nii = nb.FileHolder(fileobj=nii)
                image = nb.Nifti1Image.from_file_map({'header': nii, 'image': nii})
                array = np.asarray(image.dataobj)
        return array
    
    def mask(_file, _root: Silent, name):
        if 'Testing' in str(_file):
            image = image(_file, _root)
            return np.zeros_like(image)
        
        archive = _root / 'ATLAS_R2.0.tar.gz'
        filename = str(_file) + '_label-L_desc-T1lesion_mask.nii.gz'
        with tarfile.open(archive, "r:gz") as tar:
            nii_gz_data = tar.extractfile(filename)
                    
            with gzip.GzipFile(fileobj=nii_gz_data) as nii:
                nii = nb.FileHolder(fileobj=nii)
                mask = nb.Nifti1Image.from_file_map({'header': nii, 'image': nii})
                array = np.asarray(mask.dataobj)
        return array
    
    def affine(_file, _root: Silent):
        archive = _root / 'ATLAS_R2.0.tar.gz'
        filename = str(_file) + '_T1w.nii.gz'
        with tarfile.open(archive, "r:gz") as tar:
            nii_gz_data = tar.extractfile(filename)
                    
            with gzip.GzipFile(fileobj=nii_gz_data) as nii:
                nii = nb.FileHolder(fileobj=nii)
                image = nb.Nifti1Image.from_file_map({'header': nii, 'image': nii})
        return image.affine

ATLAS = normalize(
    ATLASBase,
    'ATLAS',
    'atlas',
    body_region='Head',
    license=licenses.CC_BY_30,
    link='http://fcon_1000.projects.nitrc.org/indi/retro/atlas.html',
    modality='MRI',
    prep_data_size='9,9G',
    raw_data_size='9,9G',
    task='Anomaly segmentation',
)