from collections import OrderedDict

from griffe.dataclasses import Alias, Class, Function, Parameters, Parameter, ParameterKind
from mkdocstrings_handlers.python.handler import PythonHandler


class PythonConnectomeHandler(PythonHandler):
    def get_templates_dir(self, handler: str):
        return super().get_templates_dir("python")

    def collect(self, identifier: str, config: dict):
        result = super().collect(identifier, config)
        if isinstance(result, Alias):
            result.target = self.patch_class(result.target)
        else:
            result = self.patch_class(result)
        return result

    def patch_class(self, x: Class):
        members = OrderedDict()
        for k, v in x.members.items():
            if not k.startswith('_'):
                assert isinstance(v, Function)
                if k == 'ids':
                    v.parameters = Parameters()
                else:
                    v.parameters = Parameters(Parameter(
                        'id', annotation='str', kind=ParameterKind.positional_or_keyword
                    ))

                members[k] = v

        x.members = members
        return x


def get_handler(theme: str, custom_templates=None, config_file_path=None, paths=None, **config):
    return PythonConnectomeHandler(
        handler="python_connectome",
        theme=theme,
        custom_templates=custom_templates,
        config_file_path=config_file_path,
        paths=paths,
    )
