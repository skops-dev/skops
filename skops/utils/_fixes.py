from types import SimpleNamespace


def boxplot(ax, *, tick_labels, **kwargs):
    """A function to handle labels->tick_labels deprecation.
    labels is deprecated in 3.9 and removed in 3.11.
    """
    try:
        return ax.boxplot(tick_labels=tick_labels, **kwargs)
    except TypeError:
        return ax.boxplot(labels=tick_labels, **kwargs)


def construct_instances(estimator):
    """Create a test instance of an estimator for compatibility testing.

    This function provides compatibility between different scikit-learn versions
    (before and after 1.6) for creating test instances of estimators. It handles
    the API change where _construct_instances was moved from estimator_checks to
    instance_generator.
    """
    try:
        from sklearn.utils._test_common.instance_generator import _construct_instances

        return next(_construct_instances(estimator))

    except ImportError:
        from sklearn.utils.estimator_checks import _construct_instance

        return _construct_instance(estimator)


def get_tags(estimator):
    """Get estimator tags in a consistent format across different sklearn versions.

    This function provides compatibility between sklearn versions before and after 1.6.
    It returns a SimpleNamespace object containing metadata about the estimator's
    requirements and capabilities (e.g., input types, fitting requirements).

    Parameters
    ----------
    estimator : estimator object
        A scikit-learn estimator instance.
    """
    try:
        from sklearn.utils._tags import get_tags

        tags = get_tags(estimator)
        return SimpleNamespace(
            input_tags=tags.input_tags,
            requires_fit=tags.requires_fit,
            pairwise=tags.input_tags.pairwise,
            two_d_array=tags.input_tags.two_d_array,
            one_d_array=tags.input_tags.one_d_array,
            three_d_array=tags.input_tags.three_d_array,
            one_d_labels=tags.target_tags.one_d_labels,
            two_d_labels=tags.target_tags.two_d_labels,
            y_required=tags.target_tags.required,
            categorical=tags.input_tags.categorical,
            dict=tags.input_tags.dict,
            string=tags.input_tags.string,
            sparse=tags.input_tags.sparse,
        )
    except ImportError:
        from sklearn.utils._tags import _safe_tags

        tags = _safe_tags(estimator)
        return SimpleNamespace(
            input_tags=tags["X_types"],
            requires_fit=tags.get("requires_fit", True),
            pairwise=tags["pairwise"],
            two_d_array="2darray" in tags["X_types"],
            one_d_array="1darray" in tags["X_types"],
            three_d_array="3darray" in tags["X_types"],
            one_d_labels="1dlabels" in tags["X_types"],
            two_d_labels="2dlabels" in tags["X_types"],
            y_required=tags["requires_y"],
            categorical="categorical" in tags["X_types"],
            dict="dict" in tags["X_types"],
            string="string" in tags["X_types"],
            sparse="sparse" in tags["X_types"],
        )
