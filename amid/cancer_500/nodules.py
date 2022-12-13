import warnings

import numpy as np

from .typing import Cancer500Nodule, Comment, Review, Texture


def get_nodules(protocol, series_number, slice_locations):
    if protocol['nodules'] is None:
        num_doctors = len(protocol['doctors'])
        assert num_doctors in [3, 6]

        if len([d for d in protocol['doctors'] if definetely_no_nodules(d['comment'])]) > num_doctors / 2:
            return []
        else:
            raise ValueError

    assert protocol['nodules']

    nodules = []
    for nodule in protocol['nodules']:
        annotations = dict(get_nodule_annotations(nodule[-1], series_number, slice_locations))
        if not annotations:
            raise ValueError

        nodules.append(annotations)

    return nodules


def definetely_no_nodules(overall_comment):
    overall_comment = overall_comment.lower()
    prefixes = ['нет очагов', 'очагов нет', 'очаги не выявлены', 'достоверно очагов нет']
    return any(overall_comment.startswith(p) for p in prefixes)


def get_nodule_annotations(nodule: dict, series_number: int, slice_locations: list):
    for rater, ann in nodule.items():
        if ann is None:
            continue

        if 'series_no' in ann and str(series_number) not in ann['series_no']:
            warnings.warn('Cannot check that annotation belongs to this particular series.')
            continue

        try:
            yield rater, parse_nodule_annotation(ann, slice_locations)
        except ValueError as e:
            warnings.warn(str(e))
            continue


def parse_nodule_annotation(ann: dict, slice_locations: list):
    return Cancer500Nodule(
        center_voxel=parse_center_voxel(ann, slice_locations),
        review=parse_review(ann),
        comment=parse_comment(ann),
        diameter_mm=parse_diameter_mm(ann),
        texture=parse_texture(ann),
        malignancy=parse_malignancy(ann),
    )


def parse_center_voxel(ann: dict, slice_locations: list):
    i, j = int(ann['x']), int(ann['y'])
    assert i == ann['x']
    assert j == ann['y']

    assert 'z type' in ann
    assert ann['z type'].strip() == 'mm'
    diff = np.abs(np.array(slice_locations) - ann['z'])
    if np.min(diff) >= 1:
        raise ValueError('Cannot determine slice.')
    slc = np.argmin(diff)

    comments = [review['comment'] for review in ann['expert decision']]
    if 'z = 258 = -151,6 ' in comments:
        slc = 258
    elif 'не 134 а 143 по оси Х' in comments:
        i = 143
    elif (
        'неправильная координата х (должно быть 73, а не 734). сосуд, несовпадение типа (другое), неверный размер'
        in comments
    ):
        i = 73
    elif 'ошибка в координате Y - должно быть 296, тогда очаг есть' in comments:
        j = 296
    elif 'срез съехал на два ниже' in comments:
        slc -= 2
    elif set(comments) & {
        'очага нет',
        'промахно',
        'промахнулись с координатой х',
        'часть координат не совпадает с топикой очага',
        'часть координат не совпадает с топикой очага, неверный размер',
    }:
        raise ValueError('Cannot detetmine center voxel')

    return i, j, slc


def parse_review(ann: dict):
    decisions = {review['decision'] for review in ann['expert decision']}
    if 'confirmed' in decisions:
        return Review.Confirmed
    elif 'confirmed_partially' in decisions:
        return Review.ConfirmedPartially
    elif 'doubt' in decisions:
        return Review.Doubt
    elif 'rejected' in decisions:
        return Review.Rejected
    else:
        raise ValueError(decisions)


def parse_comment(ann: dict):
    comments = {review['comment'] for review in ann['expert decision']}
    if 'кальцинат, несовпадение типа (другое)' in comments:
        return Comment.Calcium
    elif 'фиброз' in comments:
        return Comment.Fibrosis
    elif 'внутрилегочный л\\у' in comments:
        return Comment.LymphNode
    elif 'очаг с кальцинацией, несовпадение типа (другое)' in comments:
        return Comment.Calcified
    elif 'бронхоэктаз с содержимым, несовпадение типа (другое)' in comments:
        return Comment.Bronchiectasis
    elif 'сосуд' in comments:
        return Comment.Vessel


def parse_diameter_mm(ann: dict):
    if any('неверный размер' in review['comment'].lower() for review in ann['expert decision']):
        return

    return round(ann['diameter (mm)'], 2)


def parse_texture(ann: dict):
    nodule_types = {review['type'] for review in ann['expert decision']} & {'#0S', '#1PS', '#2GG', 'другое'}
    if nodule_types:
        assert len(nodule_types) == 1
        (nodule_type,) = nodule_types
    elif parse_review(ann) in [Review.Confirmed, Review.ConfirmedPartially, Review.Doubt]:
        assert ann['type'] in ['#0S', '#1PS', '#2GG']
        nodule_type = ann['type']
    else:
        return

    if nodule_type == '#0S':
        return Texture.Solid
    elif nodule_type == '#1PS':
        return Texture.PartSolid
    elif nodule_type == '#2GG':
        return Texture.GroundGlass
    elif nodule_type == 'другое':
        return Texture.Other


def parse_malignancy(ann: dict):
    malignant = [review['malignant'] for review in ann['expert decision']]
    if all(malignant):
        return True
    elif not any(malignant):
        return False
