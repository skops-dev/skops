.. _persistence:

Secure persistence with skops
=============================

.. warning::

   This feature is very early in development, which means the API is
   unstable and it is **not secure** at the moment. Therefore, use the same
   caution as you would for ``pickle``: Don't load from sources that you
   don't trust. In the future, more security will be added.

Skops offers a way to save and load sklearn models without using :mod:`pickle`.
Pickle is the standard serialization format for sklearn and for Python in
general. One of its biggest advantages is it can be used for almost all Python code
but this flexibility also means that it's inherently insecure. As the Python
docs say:

    The pickle module is not secure. Only unpickle data you trust. It is
    possible to construct malicious pickle data which will execute arbitrary
    code during unpickling. Never unpickle data that could have come from an
    untrusted source, or that could have been tampered with.

In contrast to pickle, the :func:`skops.io.save` and :func:`skops.io.load` 
functions cannot be used to save arbitrary Python code, but they bypass 
pickle and are thus more secure.

Usage
-----

Using :func:`skops.io.save` and :func:`skops.io.load` is quite simple. Below is
an example:

.. code:: python

    from sklearn.linear_model import LogisticRegression
    from skops.io import load, save

    clf = LogisticRegression(random_state=0, solver="liblinear")
    clf.fit(X_train, y_train)
    save(clf, "my-logistic-regression.skops")
    # ...
    loaded = load("my-logistic-regression.skops")
    loaded.predict(X_test)

At the moment, we support the vast majority of sklearn estimators. This includes
complex use cases such as :class:`sklearn.pipeline.Pipeline`,
:class:`sklearn.model_selection.GridSearchCV`, classes using Cython code, such
as :class:`sklearn.tree.DecisionTreeClassifier`, and more. If you discover an sklearn
estimator that does not work, please open an issue on the skops `GitHub page
<https://github.com/skops-dev/skops/issues>`_ and let us know.

In contrast to pickle, skops cannot persist arbitrary Python code. This means 
if you have custom functions (say, a custom function to be used 
with :class:`sklearn.preprocessing.FunctionTransformer`), it will not
work. However, most ``numpy`` and ``scipy`` functions should work. Therefore,
you can actually save built-in functions like``numpy.sqrt``.

Goals
-----

Currently, it is still possible to run insecure code when using skops
persistence. For example, it's possible to load a save file that evaluates arbitrary
code using :func:`eval`. However, we have concrete plans on how to mitigate
this, so please stay updated.

On top of trying to support all of sklearn, we plan on making persistence
extensible for other libraries. As a user, this means that if you trust a
certain library, you will be able to tell skops to load code from that library.
As a library author, there will be a clear path of what needs to be done to add
secure persistence to your library, such that skops can save and load code from
your library.

Roadmap
-------

To follow what features are currently planned, filter for the `"persistence"
label <https://github.com/skops-dev/skops/labels/persistence>`_ in our GitHub
issues.
