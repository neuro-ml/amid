from typing import NamedTuple


class ClinicalInfo(NamedTuple):
    gender: str
    age_at_scan_years: float
    survival_from_surgery_days: int
    idh1: str
    mgmt: str
    kps: str
    gtr_over90percent: str
    time_since_baseline_preop: int
    psp_tp_score: float


class AcquisitionInfo(NamedTuple):
    manufacturer: str
    model: str
    magnetic_field_strength: float
    t1_imaging_frequency: float
    t1_repetition_time: float
    t1_echo_time: float
    t1_inversion_time: float
    t1_flip_angle: float
    t1_pixel_spacing: str
    t1_slice_thickness: float
    t1gd_imaging_frequency: float
    t1gd_repetition_time: float
    t1gd_echo_time: float
    t1gd_inversion_time: float
    t1gd_flip_angle: float
    t1gd_pixel_spacing: str
    t1gd_slice_thickness: float
    t2_imaging_frequency: float
    t2_repetition_time: float
    t2_echo_time: float
    t2_flip_angle: float
    t2_pixel_spacing: str
    t2_slice_thickness: float
    flair_imaging_frequency: float
    flair_repetition_time: float
    flair_echo_time: float
    flair_inversion_time: float
    flair_flip_angle: float
    flair_pixel_spacing: str
    flair_slice_thickness: float
    dti_imaging_frequency: float
    dti_repetition_time: float
    dti_echo_time: float
    dti_flip_angle: float
    dti_pixel_spacing: str
    dti_slice_thickness: float
    dsc_imaging_frequency: float
    dsc_repetition_time: float
    dsc_echo_time: float
    dsc_flip_angle: float
    dsc_pixel_spacing: str
    dsc_slice_thickness: float
