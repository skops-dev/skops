import importlib


def _import_obj(module, cls_or_func, package=None):
    return getattr(importlib.import_module(module, package=package), cls_or_func)


def gettype(state):
    if "__module__" in state and "__class__" in state:
        return _import_obj(state["__module__"], state["__class__"])
    return None
