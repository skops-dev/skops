def boxplot(ax, *, tick_labels, **kwargs):
    """A function to handle labels->tick_labels deprecation.
    labels is deprecated in 3.9 and removed in 3.11.
    """
    try:
        return ax.boxplot(tick_labels=tick_labels, **kwargs)
    except TypeError:
        return ax.boxplot(labels=tick_labels, **kwargs)
