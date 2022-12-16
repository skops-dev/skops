class UnsupportedTypeException(TypeError):
    """Raise when an object of this type is known to be unsupported"""

    def __init__(self, obj):
        super().__init__(
            f"Objects of type {obj.__class__.__name__} are not supported yet."
        )


class UntrustedTypesFoundException(TypeError):
    """Raise when some untrusted objects are found in the file."""

    def __init__(self, unsafe):
        super().__init__(f"Untrusted types found in the file: {sorted(unsafe)}.")
