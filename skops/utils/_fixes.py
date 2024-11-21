def boxplot(ax, *, tick_labels, **kwargs):
    """A function to handle labels->tick_labels deprecation.
    labels is deprecated in 3.9 and removed in 3.11.
    """
    try:
        return ax.boxplot(tick_labels=tick_labels, **kwargs)
    except TypeError:
        return ax.boxplot(labels=tick_labels, **kwargs)


def construct_instances(estimator):
    """Added for sklearn<1.6 support."""
    try:
        from sklearn.utils._test_common.instance_generator import _construct_instances

        return next(_construct_instances(estimator))

    except ImportError:
        from sklearn.utils.estimator_checks import _construct_instances

        return _construct_instances(estimator)


def requires_fit(estimator):
    """Added for sklearn<1.6 support."""
    try:
        from sklearn.utils._tags import get_tags

        return get_tags(estimator).requires_fit
    except ImportError:
        from sklearn.utils._tags import _safe_tags

        return _safe_tags(estimator).get("requires_fit", True)
