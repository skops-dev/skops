.. _persistence:

Secure persistence with skops
=============================

.. warning::

   This feature is heavily under development, which means the API is unstable
   and there might be security issues at the moment. Therefore, use caution
   when loading files from sources you don't trust.

Skops offers a way to save and load sklearn models without using :mod:`pickle`.
The ``pickle`` module is not secure, but with skops, you can [more] securely
save and load models without using ``pickle``.

``Pickle`` is the standard serialization format for sklearn and for Python in
general (``cloudpickle`` and ``joblib`` use the same format). One of the main
advantages of ``pickle`` is that it can be used for almost all Python objects
but this flexibility also makes it inherently insecure. This is because loading
certain types of objects requires the ability to run arbitrary code, which can
be misused for malicious purposes. For example, an attacker can use it to steal
secrets from your machine or install a virus. As the `Python docs
<https://docs.python.org/3/library/pickle.html#module-pickle>`__ say:

.. warning::

    The pickle module is not secure. Only unpickle data you trust. It is
    possible to construct malicious pickle data which will execute arbitrary
    code during unpickling. Never unpickle data that could have come from an
    untrusted source, or that could have been tampered with.

In contrast to ``pickle``, the :func:`skops.io.dump` and :func:`skops.io.load`
functions have a more limited scope, while preventing users from running
arbitrary code or loading unknown and malicious objects.

When loading a file, :func:`skops.io.load`/:func:`skops.io.loads` will traverse
the input, check for known and unknown types, and will only construct those
objects if they are trusted, either by default or by the user.

.. note::
    You can try out converting your existing pickle files to the skops format
    using this Space on Hugging Face Hub:
    `pickle-to-skops <https://huggingface.co/spaces/scikit-learn/pickle-to-skops>`__.

Usage
-----

The code snippet below illustrates how to use :func:`skops.io.dump` and
:func:`skops.io.load`. Note that one needs `XGBoost
<https://xgboost.readthedocs.io/en/stable/>`__ installed to run this:

.. code:: python

    from xgboost.sklearn import XGBClassifier
    from sklearn.model_selection import GridSearchCV, train_test_split
    from sklearn.datasets import load_iris
    from skops.io import dump, load

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    param_grid = {"tree_method": ["exact", "approx", "hist"]}
    clf = GridSearchCV(XGBClassifier(), param_grid=param_grid).fit(X_train, y_train)
    print(clf.score(X_test, y_test))
    0.9666666666666667
    dump(clf, "my-model.skops")
    # ...
    loaded = load("my-model.skops", trusted=True)
    print(loaded.score(X_test, y_test))
    0.9666666666666667

    # in memory
    from skops.io import dumps, loads
    serialized = dumps(clf)
    loaded = loads(serialized, trusted=True)

Note that you should only load files with ``trusted=True`` if you trust the
source. Otherwise you can get a list of untrusted types present in the dump
using :func:`skops.io.get_untrusted_types`:

.. code:: python

    from skops.io import get_untrusted_types
    unknown_types = get_untrusted_types(file="my-model.skops")
    print(unknown_types)
    ['numpy.float64', 'numpy.int64', 'sklearn.metrics._scorer._passthrough_scorer',
    'xgboost.core.Booster', 'xgboost.sklearn.XGBClassifier']

Note that everything in the above list is safe to load. We already have many
types included as trusted by default, and some of the above values might be
added to that list in the future.

Once you check the list and you validate that everything in the list is safe,
you can load the file with ``trusted=unknown_types``:

.. code:: python

    loaded = load("my-model.skops", trusted=unknown_types)

At the moment, we support the vast majority of sklearn estimators. This
includes complex use cases such as :class:`sklearn.pipeline.Pipeline`,
:class:`sklearn.model_selection.GridSearchCV`, classes using objects defined in
Cython such as :class:`sklearn.tree.DecisionTreeClassifier`, and more. If you
discover an sklearn estimator that does not work, please open an issue on the
skops `GitHub page <https://github.com/skops-dev/skops/issues>`__ and let us
know.

At the moment, ``skops`` cannot persist arbitrary Python code. This means if
you have custom functions (say, a custom function to be used with
:class:`sklearn.preprocessing.FunctionTransformer`), it will not work. However,
most ``numpy`` and ``scipy`` functions should work. Therefore, you can save
objects having references to functions such as ``numpy.sqrt``.

Command Line Interface
######################

Skops has a command line interface to convert scikit-learn models persisted with
``Pickle`` to ``Skops`` files.

To convert a file from the command line, use the ``skops convert`` entrypoint.

Below is an example call to convert a file ``my_model.pkl`` to ``my_model.skops``:

.. code:: console

    skops convert my_model.pkl

To convert multiple files, you can use bash commands to iterate the above call.
For example, to convert all ``.pkl`` flies in the current directory:

.. code:: console

    for FILE in *.pkl; do skops convert FILE; done

Further help for the different supported options can be found by calling
``skops convert --help`` in a terminal.

Visualization
#############

Skops files can be visualized using :func:`skops.io.visualize`. If you have
a skops file called ``my-model.skops``, you can visualize it like this:

.. code:: python

    import skops.io as sio
    sio.visualize("my-model.skops")

The output could look like this:

.. code::

    root: sklearn.preprocessing._data.MinMaxScaler
    └── attrs: builtins.dict
        ├── feature_range: builtins.tuple
        │   ├── content: json-type(-555)
        │   └── content: json-type(123)
        ├── copy: unsafe_lib.UnsafeType [UNSAFE]
        ├── clip: json-type(false)
        └── _sklearn_version: json-type("1.2.0")

``unsafe_lib.UnsafeType`` was recognized as untrusted and marked.

It's also possible to visualize the object dumped as bytes:

    import skops.io as sio
    my_model = ...
    sio.visualize(sio.dumps(my_model))

There are various options to customize the output. By default, the security of
nodes is color coded if `rich <https://github.com/Textualize/rich>`_ is
installed, otherwise they all have the same color. To install ``rich``, run:

.. code::

    python -m pip install rich

or, when installing skops, install it like this:

    python -m pip install skops[rich]

To disable colors, even if ``rich`` is installed, pass ``use_colors=False`` to
:func:`skops.io.visualize`.

It's also possible to change what colors are being used, e.g. by passing
``visualize(..., color_safe="cyan")`` to change the color for trusted nodes from
green to cyan. The ``rich`` docs list the `supported standard colors
<https://rich.readthedocs.io/en/stable/appendix/colors.html>`_.

Note that the visualization feature is intended to help understand the structure
of the object, e.g. what attributes are identified as untrusted. It is not a
replacement for a proper security check. In particular, just because an object's
visualization looks innocent does *not* mean you can just call `sio.load(<file>,
trusted=True)` on this object -- only pass the types you really trust to the
``trusted`` argument.

Supported libraries
-------------------

Skops intends to support all of **scikit-learn**, that is, not only its
estimators, but also other classes like cross validation splitters. Furthermore,
most types from **numpy** and **scipy** should be supported, such as (sparse)
arrays, dtypes, random generators, and ufuncs.

Apart from this core, we plan to support machine learning libraries commonly
used be the community. So far, we have tested the following libraries:

- `LightGBM <https://lightgbm.readthedocs.io/>`_ (scikit-learn API)
- `XGBoost <https://xgboost.readthedocs.io/en/stable/>`_ (scikit-learn API)
- `CatBoost <https://catboost.ai/en/docs/>`_

If you run into a problem using any of the mentioned libraries, this could mean
there is a bug in skops. Please open an issue on `our issue tracker
<https://github.com/skops-dev/skops/issues>`__ (but please check first if a
corresponding issue already exists).

In terms of security, we do not audit these libraries for security issues.
Therefore, you should only load a skops file containing a model of any of those
libraries if you trust them to be secure. It's not a perfect solution, but it's
still better than trusting pickle files, which anyone can tamper with easily.

Backwards compatibility
-----------------------

Compatibility across skops versions
===================================

The skops persistence format is in flux, as we steadily work on improving it,
making it more secure and supporting more types. When we make a change that is
incompatible with existing skops files, the protocol will be bumped to the next
higher number (the protocol can be checked in the schema of the skops file). At
the same time, we will ensure that existing skops files with lower protocol
versions still load as always, even if they contained a bug (in which case we
will warn about it). Therefore, it is generally safe to assume that your skops
files will keep on working with future versions of skops.

You may want to periodically load and dump old skops files using newer versions
of skops to benefit from the updates to the protocol.

One caveat to the backwards compatibility promise is that the skops files have
to be created based on a release version of skops. If you create skops files
using a skops version installed from the main branch, it is possible to end up
in an inconsistent state. Therefore, don't use the main branch of skops for
creating skops files intended to be loaded with future skops versions.

Compatibility across sklearn versions
=====================================

Using skops to load a model saved in one sklearn version and loading it with
another sklearn version is not recommended, because the behavior of the model
may change across versions. In some cases loading the model in a different
version might not be possible due to internal changes in scikit-learn. Such
changes don't happen very often, but they can happen, thus you should be
cautious. To replicate a model trained with one sklearn version using a
different sklearn version, it is advised to retrain the model on the same data
using the same training process.

The potential compatibility issue between sklearn versions is not skops
specific. It is general sklearn behavior which skops cannot avoid. According to
the sklearn `docs on model persistence
<https://scikit-learn.org/stable/model_persistence.html#security-maintainability-limitations>`_:

    While models saved using one version of scikit-learn might load in other
    versions, this is entirely unsupported and inadvisable. It should also be
    kept in mind that operations performed on such data could give different and
    unexpected results.

Roadmap
-------

There needs to be more testing to harden the loader and make sure we don't run
arbitrary code when it's not intended. However, the safety mechanisms already
in place should prevent most cases of abuse.

At the moment, persisting and loading arbitrary C extension types is not
possible, unless a python object wraps around them and handles persistence and
loading via ``__getstate__`` and ``__setstate__``. We plan to develop an API
which would help third party libraries to make their C extension types
``skops`` compatible.

You can check on our `"issue tracker
<https://github.com/skops-dev/skops/labels/persistence>`__ which features are
planned for the near future.
