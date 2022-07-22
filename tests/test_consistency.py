from amid.internals import gather_datasets


def test_ids_availability():
    for cls, _, _ in gather_datasets().values():
        assert len(cls().ids) > 0
