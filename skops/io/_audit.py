import warnings
from functools import wraps
from typing import Any, Callable, Literal, Protocol, Sequence

from .exceptions import InsecureObjectException


def _get_func_name(state, module=None):
    """Given a state dict, return function name

    Optionally, also indicate the module of said function. Works on nested
    states. If the name cannot be determined, return "unknown".

    Examples
    --------
    >>> import numpy as np
    >>> from functools import partial
    >>> from skops.io._persist import get_state
    >>> from sklearn.metrics import accuracy_score
    >>> _get_func_name({})
    "'unknown'"
    >>> _get_func_name(get_state(123, None))
    "'unknown'"
    >>> _get_func_name(get_state(accuracy_score, None))
    "'accuracy_score' of module 'sklearn.metrics._classification'"
    >>> _get_func_name(get_state(np.add, None))
    "'add' of module 'numpy'"
    >>> _get_func_name(get_state(partial(np.add, 3), None))
    "'add' of module 'numpy'"
    """
    # known limitation:
    # _get_func_name(get_state(scipy.special.beta, None)) == "'beta' of module 'numpy'"
    if isinstance(state, str):
        # found the function name, finish recursing
        return f"'{state}' of module '{module}'" if module else f"'{state}'"

    content = state.get("content", {})
    if not content or not isinstance(content, dict):
        return "'unknown'"

    # get name for functions or partials
    func = content.get("function") or content.get("func")
    module = state.get("__module__")
    return _get_func_name(func, module=module)


class Audit(Protocol):
    # define a protocol so that both functions and callables are accepted
    def __call__(self, state: dict[str, Any]) -> tuple[bool, str]:
        ...


def audit_state_sanity(state: dict[str, Any]) -> tuple[bool, str]:
    """Perform sanity checks on the state"""
    if not isinstance(state, dict):
        raise TypeError("state must be a dict")

    try:
        _, _ = state["__module__"], state["__class__"]
    except Exception:
        # can't figure out the type, so it's considered insecure
        return False, "State must contain keys '__module__' and '__class__'"

    return True, ""


def audit_is_function(state: dict[str, Any]) -> tuple[bool, str]:
    """Check whether a function is being loaded"""
    if state["__class__"] == "function" or state["__class__"] == "partial":
        func_name = _get_func_name(state)
        msg = f"Loading function '{func_name}' is considered insecure"
        return False, msg
    return True, ""


class AuditSecureModule:
    """Check whether an object being loaded comes from a secure module

    Parameters
    ----------
    secure_models: sequence of str or literal "*"
        Name of the modules to be considered as secure, e.g. ``["numpy",
        "sklearn"]``. It is possible to add a wildcard string ``"*"``, which
        results in all modules being accepted as secure.

    """

    def __init__(self, secure_modules: Sequence[str] | Literal["*"]):
        self.secure_modules = secure_modules

    def __call__(self, state: dict[str, Any]) -> tuple[bool, str]:
        if self.secure_modules == "*":
            return True, ""

        module = state["__module__"].split(".", 1)[0]
        if module not in self.secure_modules:
            return False, f"Untrusted module '{module}' found"

        return True, ""


class AuditChain:
    """Chain audit functions and callables together and wraps ``get_instance``

    After initializing this class and wrapping each ``get_instance`` method with
    it, when the ``get_instance`` method is being called, an audit of the
    ``state`` will be performed before actually loading the object. This way, it
    can be ensured that the object is secure.

    Parameters
    ----------
    audit_hooks : sequence of callables
        All audits that should be called on the state before any objects are
        being loaded. Audit functions are called recursively on the object to be
        loaded and all its attributes and sub-attributes. An audit function
        should accept a dict (the state) as input and return a tuple. The first
        element should be a bool of whether the object should be considered
        secure or not. The second is a message str to tell the user why the
        object is considered insecure -- if the object is secure, just return an
        empty string.

    method : str or callable (default="raise")
        The method of how to deal with insecure objects. If a string:

        - "raise": Raises a skops.io.exceptios.InsecureObjectException
        - "warn": Warn but still load the object
        - "print": Only print a message
        - "ignore": Completely ignore insecure objects

        If a callable, it should accept the message generated regarding the
        insecure object. It doesn't need to have a return message. As an
        example, you could set ``method=logger.warning`` if you want to have a
        warning about potentially insecure objects but still allow to load them.

    """

    def __init__(
        self,
        audit_hooks: Sequence[Audit],
        method: str | Callable[[str], None] = "raise",
    ):
        self.audit_hooks = audit_hooks

        if isinstance(method, str):
            method = method.lower()
            methods_allowed = ("raise", "warn", "print", "ignore")
            if method not in methods_allowed:
                raise ValueError(
                    f"method {method} should be one of {methods_allowed} or a function"
                )

        self.method = method

    def __call__(self, func):
        """Decorator that performs audits before executing the function"""

        @wraps(func)
        def wrapper(state, *args, **kwargs):
            # logic for collecting audits
            audits_insecure = []
            for audit in self.audit_hooks:
                is_secure, msg = audit(state)
                if not is_secure:
                    audits_insecure.append((is_secure, msg))

            # all good, proceed to load object
            if not audits_insecure:
                return func(state, *args, **kwargs)

            # at least one security issue found
            if len(audits_insecure) == 1:
                words = "security violation has"
            else:
                words = f"{len(audits_insecure)} security violations have"
            msg = f"The following {words} been found: "
            msg = msg + ". ".join(msg for _, msg in audits_insecure)

            if self.method == "raise":
                raise InsecureObjectException(msg)

            if self.method == "warn":
                warnings.warn(msg, UserWarning)
            elif self.method == "print":
                print("\n!!! " + msg)
            elif callable(self.method):
                self.method(msg)

            return func(state, *args, **kwargs)

        return wrapper
