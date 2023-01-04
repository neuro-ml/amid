from .const import COLUMN2LABEL


def add_labels(scope):
    def make_loader(column):
        def loader(i, _meta):
            # ambigous data in meta
            if int(i) in [500, 600]:
                return None

            return _meta[_meta['amos_id'] == int(i)][column].item()

        return loader

    for column, label in COLUMN2LABEL.items():
        scope[label] = make_loader(column)
