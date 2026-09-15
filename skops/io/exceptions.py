class UnsupportedTypeException(TypeError):
    """Raise when an object of this type is known to be unsupported"""

    def __init__(self, obj):
        super().__init__(
            f"Objects of type {obj.__class__.__name__} are not supported yet."
        )


# Types that are known to be unsafe to trust by default, together with a
# human-readable explanation of *why*. skops can only verify that a loaded
# object is of one of these types, not that its content is safe: a crafted
# file can still pass the type check and crash the process (or worse) once
# the resulting object is used. Shown to users when one of these types is
# flagged as untrusted. See https://github.com/scikit-learn/scikit-learn/pull/34558.
UNTRUSTED_TYPE_REASONS: dict[str, str] = {
    "sklearn.tree._tree.Tree": (
        "sklearn.tree._tree.Tree (the shared node storage for DecisionTree*, "
        "RandomForest*, ExtraTrees*, and GradientBoosting* models) stores raw "
        "node indices (left_child, right_child, feature) that scikit-learn "
        "indexes into without bounds checking. A malicious file can set these "
        "to out-of-range values: skops loads the object successfully, but "
        "calling .predict() on it can then crash the process (segfault) or "
        "read out-of-bounds memory. If you created the file yourself or "
        "otherwise fully trust its source, you can load it with "
        'trusted=["sklearn.tree._tree.Tree"].'
    ),
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor": (
        "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor "
        "(used by HistGradientBoosting* models) stores raw node indices "
        "(left, right, feature_idx) in its nodes array that scikit-learn "
        "indexes into without bounds checking. A malicious file can set "
        "these to out-of-range values: skops loads the object "
        "successfully, but calling .predict() on it can then crash the "
        "process (segfault) or read out-of-bounds memory. If you created "
        "the file yourself or otherwise fully trust its source, you can "
        "load it with trusted=["
        '"sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"].'
    ),
}


class UntrustedTypesFoundException(TypeError):
    """Raise when some untrusted objects are found in the file."""

    def __init__(self, unsafe):
        unsafe = sorted(unsafe)
        msg = f"Untrusted types found in the file: {unsafe}."
        notes = [
            f"- {name}: {UNTRUSTED_TYPE_REASONS[name]}"
            for name in unsafe
            if name in UNTRUSTED_TYPE_REASONS
        ]
        if notes:
            msg += "\n\n" + "\n\n".join(notes)
            msg += (
                "\n\nOnly add the specific types you have reviewed and trust "
                "to the `trusted` argument; avoid passing everything "
                "reported by get_untrusted_types() just to make a file load."
            )
        super().__init__(msg)
