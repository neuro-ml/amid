fields = [
    'site_id',
    'patient_id',
    'image_id',
    'laterality',
    'view',
    'age',
    'cancer',
    'biopsy',
    'invasive',
    'BIRADS',
    'implant',
    'density',
    'machine_id',
    'prediction_id',
    'difficult_negative_case',
]


def add_csv_fields(scope):
    def make_loader(field):
        def loader(_row):
            return getattr(_row, field, None)

        return loader

    for _field in fields:
        scope[_field] = make_loader(_field)
