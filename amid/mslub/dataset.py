from pathlib import Path

import nibabel as nb

from ..internals import Dataset, licenses, register


@register(
    body_region='Head',
    license=licenses.CC_BY_30,
    link='https://github.com/muschellij2/open_ms_data?tab=readme-ov-file',
    modality='MRI',
    prep_data_size='18G',
    raw_data_size='5.9G',
    task='Anomaly segmentation',
)
class MSLUB(Dataset):
    @property
    def ids(self):
        result = set()
        for file in self.root.glob('**/*.gz'):
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

    def _file(self, i):
        plane = i.split('-')[0]
        patient = i.split('-')[1]
        path = self.root / plane / 'raw' / patient
        if 'longitudinal' in i:
            study_number = i.split('-')[2]
            return path / study_number
        return path

    def image(self, i):
        file = self._file(i)
        if 'longitudinal' in str(file):
            study_number = file.stem
            file_name = file.parent / f'{study_number}_FLAIR.nii.gz'
        else:
            file_name = file / 'FLAIR.nii.gz'
        image = nb.load(file_name).get_fdata()
        return image

    def mask(self, i):
        file = self._file(i)
        if 'longitudinal' in str(file):
            file_name = file.parent / 'gt.nii.gz'
        else:
            file_name = file / 'consensus_gt.nii.gz'
        image = nb.load(file_name).get_fdata()
        return image

    def patient(self, i):
        file = self._file(i)
        if 'longitudinal' in str(file):
            return Path(file).parent.name
        else:
            return Path(file).name

    def affine(self, i):
        file = self._file(i)
        if 'longitudinal' in str(file):
            study_number = file.stem
            file_name = file.parent / f'{study_number}_FLAIR.nii.gz'
        else:
            file_name = file / 'FLAIR.nii.gz'
        return nb.load(file_name).affine
