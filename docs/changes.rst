.. include:: _authors.rst

.. _changelog:

skops Changelog
===============

.. contents:: Table of Contents
    :depth: 1
    :local:


v0.9
----
- Add support for `quantile-forest <https://github.com/zillow/quantile-forest>`__
  estimators. :pr:`384` by :user:`Reid Johnson <reidjohnson>`.
- Fix an issue with visualizing Skops files for `scikit-learn` tree estimators.
  :pr:`386` by :user:`Reid Johnson <reidjohnson>`.

v0.8
----
- Adds the abillity to set the :attr:`.Section.folded` property when using :meth:`.Card.add`.
  :pr:`361` by :user:`Thomas Lazarus <lazarust>`.
- Add the CLI command to update Skops files to the latest Skops persistence format.
  (:func:`.cli._update.main`). :pr:`333` by :user:`Edoardo Abati <EdAbati>`
- Fix a bug that prevented persisting ``np.mean`` when using numpy>=1.25.0.
  :pr:`373` by `Adrin Jalali`_.

v0.7
----
- Add ability to copy plots on :meth:`.Card.save` so that they can be
  referenced in the model card. :pr:`330` by :user:`Thomas Lazarus <lazarust>`.
- `compression` and `compresslevel` from :class:`~zipfile.ZipFile` are now
  exposed to the user via :func:`.io.dumps` and :func:`.io.dump`. :pr:`345` by
  `Adrin Jalali`_.
- Fix: :func:`skops.io.visualize` is now capable of showing bytes. :pr:`352` by
  `Benjamin Bossan`_.
- All public ``numpy`` ufuncs (Universal Functions) and dtypes are trusted by default
  by :func:`.io.load`. :pr:`336` by :user:`Omar Arab Oghli <omar-araboghli>`.
- Sections in :class:`skops.card.Card` can now be folded once added to the
  card. :pr:`341` by :user:`Thomas Lazarus <lazarust>`.
- Model loading in :class:`skops.card.Card` is now cached to avoid loading the
  model multiple times. :pr:`299` by :user:`Juan Camacho <jucamohedano>`.

v0.6
----
- Add tabular regression example. :pr:`254` by :user:`Thomas Lazarus <lazarust>`.
- All public ``scipy.special`` ufuncs (Universal Functions) are trusted by default
  by :func:`.io.load`. :pr:`295` by :user:`Omar Arab Oghli <omar-araboghli>`.
- Add a new function :func:`skops.card.Card.add_metric_frame` to help users
  add metrics to their model cards. :pr:`298` by :user:`Thomas Lazarus <lazarust>`
- Add :func:`.Card.create_toc` to create a table of contents for the model card in
  markdown format. :pr:`305` by :user:`Thomas Lazarus <lazarust>`.
- Add example of using model card without the skops template. :pr:`291` by
  `Benjamin Bossan`_.
- Fix: skops persistence now also works with many functions from the
  :mod:`operator` module. :pr:`287` by `Benjamin Bossan`_.
- ``add_*`` methods on :class:`.Card` now have default section names (but
  ``None`` is no longer valid) and no longer add descriptions by default.
  :pr:`321` by `Benjamin Bossan`_.
- Add possibility to visualize a skops object and show untrusted types by using
  :func:`skops.io.visualize`. For colored output, install `rich`: `pip install
  rich`. :pr:`317` by `Benjamin Bossan`_.
- Fix issue with persisting :class:`numpy.random.Generator` using the skops
  format (the object could be loaded correctly but security could not be
  checked). :pr:`331` by `Benjamin Bossan`_.

v0.5
----
- Add CLI entrypoint support (:func:`.cli.entrypoint.main_cli`)
  and a command line function to convert Pickle files
  to Skops files (:func:`.cli._convert.main`). :pr:`249` by `Erin Aho`_
- Support more array-like data types for tabular data and list-like data types
  for text data. :pr:`179` by :user:`Francesco Cariaggi <anferico>`.
- Add an option `use_intelex` to :func:`skops.hub_utils.init` which, when
  enabled, will result in the Hugging Face inference API running with Intel's
  scikit-learn intelex library, which can accelerate inference times. :pr:`267`
  by `Benjamin Bossan`_.
- Model cards that have been written into a markdown file can now be parsed back
  into a :class:`skops.card.Card` object and edited further by using the
  :func:`skops.card.parse_modelcard` function. :pr:`257` by `Benjamin Bossan`_.

v0.4
----
- :func:`.io.dump` and :func:`.io.load` now work with file like objects,
  which means you can use them with the ``with open(...) as f: dump(obj, f)``
  pattern, like you'd do with ``pickle``. :pr:`234` by `Benjamin Bossan`_.
- All `scikit-learn` estimators are trusted by default.
  :pr:`237` by :user:`Edoardo Abati <EdAbati>`.
- Add `model_format` argument to :meth:`skops.hub_utils.init` to be stored in
  `config.json` so that we know how to load a model from the repository.
  :pr:`242` by `Merve Noyan`_.
- Persistence now supports bytes and bytearrays, added tests to verify that
  LightGBM, XGBoost, and CatBoost work now. :pr:`244` by `Benjamin Bossan`_.
- :class:`.card.Card` now allows to add content to existing sections, using a
  ``/`` to separate the subsections. E.g. use ``card.add(**{"Existing
  section/New section": "content"})`` to add "content" a new subsection called
  "New section" to an existing section called "Existing section". :pr:`203` by
  `Benjamin Bossan`_.

v0.3
----
- Utility function to add arbitrary files to be uploaded to the hub by using
  :func:`.hub_utils.add_files`. :pr:`123` by `Benjamin Bossan`_.
- Add ``private`` as an optional argument to :meth:`skops.hub_utils.push` to
  optionally set the visibility status of a repo when pushing to the hub.
  :pr:`130` by `Adrin Jalali`_.
- First release of the skops secure persistence feature (:pr:`128`) by `Adrin
  Jalali`_ and `Benjamin Bossan`_. Visit :ref:`persistence` for more
  information. This feature is not production ready yet but we're happy to
  receive feedback from users.
- Fix a bug that resulted in markdown tables being rendered incorrectly if
  entries contained line breaks. :pr:`156` by `Benjamin Bossan`_.
- Raise an error instead of warning the user if a given model file is empty.
  :pr:`214` by `Adrin Jalali`_.
- Use ``huggingface_hub`` v0.10.1 for model cards, drop ``modelcards``
  dependency. :pr:`162` by `Benjamin Bossan`_.
- Add source links to API documentation. :pr:`172` by :user:`Ayyuce Demirbas
  <ayyucedemirbas>`.
- Add support to load model if given Path/str to ``model`` argument in
  :mod:`skops.card` . :pr:`205` by :user:`Prajjwal Mishra <p-mishra1>`.


v0.2
----
- Tables, e.g. cross-validation results, can now be added to model cards using
  the :meth:`.Card.add_table` method. :pr:`90` by `Benjamin Bossan`_.
- Add method :meth:`.Card.render` which returns the model card as a string.
  :pr:`94` by `Benjamin Bossan`_.
- Make :meth:`skops.hub_utils.init` atomic. Now it doesn't leave a trace on the
  filesystem if it fails for some reason. :pr:`60` by `Adrin Jalali`_
- When adding figures or tables, it's now possible to set ``folded=True`` to
  render the content inside a details tag. :pr:`108` by `Benjamin Bossan`_.
- Add :meth:`skops.hub_utils.get_model_output` to get the model's output using
  The Hugging Face Hub's inference API, and return an array with the outputs.
  :pr:`105` by `Adrin Jalali`_.

v0.1
----

This is the first release of the library. It include two main modules:

- :mod:`skops.hub_utils`: tools to create a model repository to be stored on
  `Hugging Face Hub <https://hf.co/models>`__, mainly through
  :func:`skops.hub_utils.init` and :func:`skops.hub_utils.push`.
- :mod:`skops.card`: tools to create a model card explaining what the model does
  and how it should be used. The model card can then be stored as the
  ``README.md`` file on the Hugging Face Hub, with pre-populated metadata to
  help Hub understand the model.


Contributors
~~~~~~~~~~~~

:user:`Adrin Jalali <adrinjalali>`, :user:`Merve Noyan <merveenoyan>`,
:user:`Benjamin Bossan <BenjaminBossan>`, :user:`Ayyuce Demirbas
<ayyucedemirbas>`, :user:`Prajjwal Mishra <p-mishra1>`, :user:`Francesco Cariaggi <anferico>`,
:user:`Erin Aho <E-Aho>`, :user:`Thomas Lazarus <lazarust>`
