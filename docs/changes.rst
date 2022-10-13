.. include:: _authors.rst

.. _changelog:

skops Changelog
===============

.. contents:: Table of Contents
    :depth: 1
    :local:

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
- Use ``huggingface_hub`` v0.10.1 for model cards, drop ``modelcards``
  dependency. :pr:`162` by `Benjamin Bossan`_.
- Add source links to API documentation. :pr:`172` by `Ayyuce Demirbas`_.


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
:user:`Benjamin Bossan <BenjaminBossan>`, :user:`Ayyuce Demirbas <ayyucedemirbas>`
