from pathlib import Path

import nibabel as nb
from connectome import Source, meta
from connectome.interface.nodes import Silent

from ..internals import licenses, normalize


class MSLUBBase(Source):
    _root: str = None

    @meta
    def ids(_root: Silent):
        result = set()
        for file in Path(_root).glob('**/*.gz'):
            if ('raw' not in str(file)) or ('gt' in str(file)):
                continue
            patient = file.parent.name
            plane = file.parent.parent.parent.name
            ind = f'{plane}-{patient}'
            if 'longitudinal' in str(file):
                filename = file.name
                study_number = filename.split('_')[0]
                ind = f'{ind}-{study_number}'
            result.add(ind)
        return list(result)

    def _file(i, _root: Silent):
        plane = i.split('-')[0]
        patient = i.split('-')[1]
        path = Path(_root) / plane / 'raw' / patient
        if 'longitudinal' in i:
            study_number = i.split('-')[2]
            return path / study_number
        return path

    def image(_file):
        if 'longitudinal' in str(_file):
            study_number = _file.stem
            file_name = _file.parent / f'{study_number}_FLAIR.nii.gz'
        else:
            file_name = _file / 'FLAIR.nii.gz'
        image = nb.load(file_name).get_fdata()
        return image

    def mask(_file):
        if 'longitudinal' in str(_file):
            file_name = _file.parent / 'gt.nii.gz'
        else:
            file_name = _file / 'consensus_gt.nii.gz'
        image = nb.load(file_name).get_fdata()
        return image

    def patient(_file):
        if 'longitudinal' in str(_file):
            return Path(_file).parent.name
        else:
            return Path(_file).name

    def affine(_file):
        if 'longitudinal' in str(_file):
            study_number = _file.stem
            file_name = _file.parent / f'{study_number}_FLAIR.nii.gz'
        else:
            file_name = _file / 'FLAIR.nii.gz'
        return nb.load(file_name).affine


MSLUB = normalize(
    MSLUBBase,
    'MSLUB',
    'mslub',
    body_region='Head',
    license=licenses.CC_BY_30,
    link='https://github.com/muschellij2/open_ms_data?tab=readme-ov-file',
    modality='MRI',
    prep_data_size='18G',
    raw_data_size='5.9G',
    task='Anomaly segmentation',
)
