.. _persistence:

Secure persistence with skops
=============================

.. warning::

   This feature is very early in development, which means the API is
   unstable and it is **not secure** at the moment. Therefore, use the same
   caution as you would for ``pickle``: Don't load from sources that you
   don't trust. In the future, more security will be added.

Skops offers a way to save and load sklearn models without using :mod:`pickle`.
The ``pickle`` module is not secure, but with skops, you can securely save and
load sklearn models without using ``pickle``.

``Pickle`` is the standard serialization format for sklearn and for Python in
general. One of the main advantages of ``pickle`` is that it can be used for
almost all Python code but this flexibility also makes it inherently insecure.
This is because loading certain types of objects requires the ability to run
arbitrary code, which can be misused for malicious purposes. For example, an
attacker can use it to steal secrets from your machine or install a virus. As
the `Python docs
<https://docs.python.org/3/library/pickle.html#module-pickle>`__ say:

.. warning::

    The pickle module is not secure. Only unpickle data you trust. It is
    possible to construct malicious pickle data which will execute arbitrary
    code during unpickling. Never unpickle data that could have come from an
    untrusted source, or that could have been tampered with.

In contrast to ``pickle``, the :func:`skops.io.dump` and :func:`skops.io.load`
functions cannot be used to save arbitrary Python code, but they bypass
``pickle`` and are thus more secure.

Usage
-----

The code snippet below illustrates how to use :func:`skops.io.dump` and
:func:`skops.io.load`:

.. code:: python

    from sklearn.linear_model import LogisticRegression
    from skops.io import dump, load

    clf = LogisticRegression(random_state=0, solver="liblinear")
    clf.fit(X_train, y_train)
    dump(clf, "my-logistic-regression.skops")
    # ...
    loaded = load("my-logistic-regression.skops", trusted=True)
    loaded.predict(X_test)

    # in memory
    from skops.io import dumps, loads
    serialized = dumps(clf)
    loaded = loads(serialized, trusted=True)

Note that you should only load files with ``trusted=True`` if you trust the
source. Otherwise you can get a list of untrusted types present in the dump
using :func:`skops.io.get_untrusted_types`:

.. code:: python

    from skops.io import get_untrusted_types
    untrusted_types = get_untrusted_types("my-logistic-regression.skops")
    print(untrusted_types)

Once you check the list and you validate that everything in the list is safe,
you can load the file with ``trusted=untrusted_types``:

.. code:: python

    loaded = load("my-logistic-regression.skops", trusted=untrusted_types)

At the moment, we support the vast majority of sklearn estimators. This
includes complex use cases such as :class:`sklearn.pipeline.Pipeline`,
:class:`sklearn.model_selection.GridSearchCV`, classes using Cython code, such
as :class:`sklearn.tree.DecisionTreeClassifier`, and more. If you discover an
sklearn estimator that does not work, please open an issue on the skops `GitHub
page <https://github.com/skops-dev/skops/issues>`_ and let us know.

In contrast to ``pickle``, skops cannot persist arbitrary Python code. This
means if you have custom functions (say, a custom function to be used with
:class:`sklearn.preprocessing.FunctionTransformer`), it will not work. However,
most ``numpy`` and ``scipy`` functions should work. Therefore, you can actually
save built-in functions like``numpy.sqrt``.

Roadmap
-------

Currently, it is still possible to run insecure code when using skops
persistence. For example, it's possible to load a save file that evaluates
arbitrary code using :func:`eval`. However, we have concrete plans on how to
mitigate this, so please stay updated.

On top of trying to support persisting all relevant sklearn objects, we plan on
making persistence extensible for other libraries. As a user, this means that
if you trust a certain library, you will be able to tell skops to load code
from that library. As a library author, there will be a clear path of what
needs to be done to add secure persistence to your library, such that skops can
save and load code from your library.

To follow what features are currently planned, filter for the `"persistence"
label <https://github.com/skops-dev/skops/labels/persistence>`_ in our GitHub
issues.
