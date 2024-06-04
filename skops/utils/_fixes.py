def boxplot(ax, *, tick_labels, **kwargs):
    """A function to handle labels->tick_labels deprecation."""
    try:
        return ax.boxplot(tick_labels=tick_labels, **kwargs)
    except TypeError:
        return ax.boxplot(labels=tick_labels, **kwargs)
