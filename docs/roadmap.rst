.. _roadmap:

Project's Roadmap
=================

``skops`` is a project which deals with certain challenges related to
operationalizing scikit-learn based models. To that end, the following areas
are in our focus, and are already implemented or will be in worked on in the
near future.

Model Cards
-----------

We have now :ref:`tools <model_card>` to generate model cards, but there
are still some rough edges which need to be smoothed out. Model cards have
the potential to be adopted by upstream scikit-learn, which would be
possible since we have almost no external dependency there. You can see the
list of open issues on model cards `here
<https://github.com/skops-dev/skops/issues?q=is%3Aissue+is%3Aopen+label%3A%22model+cards%22+>`__.
In particular:

- An app to create or modify a model card: This would also include easy
  ways for users to add inspection of the models to the card. They can
  potentially upload the model and the data, and easily generate good
  visualizations about different aspects of the model. This work is started
  in :pr:`307`.
- There are issues with model cards that make it a bit tricky to work with
  them, such as linking images. These issues need to be resolves so that
  new users don't encounter too many hurdles when they first start using
  the tools.

Persistence
-----------
When deploying models, persistence is a key aspect. Since :ref:`pickles are
insecure <persistence>`, we helps users in two ways to replace pickle
files. One is our own ``.skops`` format, and another one is through tools
to make it easier to convert models to ONNX. There are existing tools to
`convert models to ONNX <http://onnx.ai/sklearn-onnx/>`__, but there are
challenges with them, and we plan to make that experience better. Some of
the work in this front will stay in this library, and some will move
upstream.

- skops: The format is in a good shape, and has been easy to work with in our
  experiments. It also supports a wide range of models, including non
  scikit-learn models. However, it requires more work to be considered more
  stable and ready for production in a larger scale. The issues can be found
  `here
  <https://github.com/skops-dev/skops/issues?q=is%3Aissue+is%3Aopen++label%3Apersistence+>`__.
  In particular:

  - We need to better be able to inspect a given file. Right now only a set of
    unknown types are given to the user, and it's not clear where they're used.

  - We need to allow support for custom c-extension types, as well as allowing
    other libraries to extend the functionality.

  - There are many optimization potentials on the format's speed and size.

  - this is high priority if we want to push it more aggressively. For
    instance, it would need to be more stable for a place like sagemaker to
    potentially support it.

- ONNX: This format is already in use by many. The `pypi download stats
  <https://pypistats.org/packages/skl2onnx>`__ at the time of writing this
  document shows 20k-30k downloads per day. However, only simple cases and not
  all estimators are supported. Right now, we haven't started working on this
  front in ``skops``, but once we do, here are some aspects to tackle:

  - Better tools to check if user's model is supported, and to add out of the
    box support for complex estimators such as pipelines and column
    transformers.

  - We should figure out how to document the process of writing a converter for
    a custom estimators. Advanced users almost always use estimators which are
    not in scikit-learn itself, and they'd need to be able to convert those
    estimators if they are to use ONNX.

Serving
-------
A very important aspect of putting a model in production, is serving the model.
There are many different ways to do that, and the right solution depends on
many parameters related to the infrastructure in use. However, some of us
maintain the relevant parts on the Hugging Face Hub to serve scikit-learn
models under the `api-inference-community repo
<https://github.com/huggingface/api-inference-community>`__. There are issues
with the current implementation, which would need some work, namely:

- We do serving right now, but it's very slow, and half the time gives a
  timeout.
- There issues related to specific dtypes, and conversion from different
  tabular formats (pandas, numpy, etc.)
- The backend could support a better way than sending/receiving json.
- There are potentials for improving inference performance, by using mkl for
  example.

Note that we're not sure about the priority of the above issues, since that
backend has little usage. But it's more of a chicken and egg problem, and if it
was to be faster, people might use it, or the tech behind it.

We can also document a simple way to serve models using one technology such as
`fastapi <https://fastapi.tiangolo.com/>`__. This would be a good start for
many developers who are new to serving their models.
