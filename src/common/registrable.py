from collections import defaultdict
from typing import Optional, TypeVar, Tuple, Dict, Type, Callable, cast

_T = TypeVar("_T")
_RegistrableT = TypeVar("_RegistrableT", bound="Registrable")

_SubclassRegistry = Dict[str, Tuple[type, Optional[str]]]


class Registrable(object):
    """
    A class that collects all registered components,
    adapted from `common.registrable.Registrable` from AllenNLP
    """
    _registered_components = defaultdict(dict)

    @classmethod
    def register(cls, name, override=False):
        registry = Registrable._registered_components[cls]

        def register_class(subclass: Type[_T]) -> Type[_T]:
            if name in registry and not override:
                raise RuntimeError('class %s already registered' % name)

            registry[name] = subclass
            return subclass

        return register_class

    @classmethod
    def by_name(cls: Type[_RegistrableT], name) -> Callable:
        return Registrable._registered_components[cls][name]

    def list_available(self):
        return list(self._registered_components)
