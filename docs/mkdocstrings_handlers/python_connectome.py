import importlib
from collections import OrderedDict

from griffe.dataclasses import Alias, Attribute, Class, Function, Parameter, ParameterKind, Parameters
from mkdocstrings_handlers.python.handler import PythonHandler


class PythonConnectomeHandler(PythonHandler):
    def get_templates_dir(self, handler: str):
        return super().get_templates_dir('python')

    def collect(self, identifier: str, config: dict):
        result = super().collect(identifier, config)
        m, p = result.path.rsplit('.', 1)
        v = getattr(importlib.import_module(m), p)
        if hasattr(v, '__origin__'):
            origin = v.__origin__
            if origin.__qualname__ != result.name:
                origin = super().collect(f'{origin.__module__}.{origin.__qualname__}', config)
                origin.name = result.name
                result = origin

        if isinstance(result, Alias):
            result.target = self.patch_class(result.target)
        else:
            result = self.patch_class(result)
        return result

    @staticmethod
    def patch_class(x: Class):
        members = OrderedDict()
        for name, v in x.members.items():
            if not name.startswith('_'):
                if isinstance(v, Function):
                    if name == 'ids':
                        v.parameters = Parameters()
                    else:
                        v.parameters = Parameters(
                            Parameter('id', annotation='str', kind=ParameterKind.positional_or_keyword)
                        )

                elif isinstance(v, Attribute):
                    v = Function(
                        name,
                        parameters=Parameters(
                            Parameter('id', annotation='str', kind=ParameterKind.positional_or_keyword)
                        ),
                        parent=x,
                    )

                else:
                    raise TypeError(v)

                members[name] = v

        x.members = members
        return x


def get_handler(theme: str, custom_templates=None, config_file_path=None, paths=None, **config):
    return PythonConnectomeHandler(
        handler='python_connectome',
        theme=theme,
        custom_templates=custom_templates,
        config_file_path=config_file_path,
        paths=paths,
    )
