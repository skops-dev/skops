.. skops documentation master file, created by
   sphinx-quickstart on Thu May  5 11:43:45 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to skops's documentation!
=================================

``skops`` is a Python library helping you share your `scikit-learn
<https://scikit-learn.org/stable/>`__ based models and put them in production.

The library is still a work in progress and under active development. You can
find the source code and the development discussions on `Github
<https://github.com/skops-dev/skops>`__.

The following examples are good starting points:

- How to create and initialize a scikit-learn model repo:
  :ref:`sphx_glr_auto_examples_plot_hf_hub.py`. You can see all the models
  uploaded to the Hugging Face Hub using this library `here
  <https://huggingface.co/models?other=skops>`_.
- How to create a model card for your scikit-learn based model:
  :ref:`sphx_glr_auto_examples_plot_model_card.py`
- A text classification example, and its integration with the hub:
  :ref:`sphx_glr_auto_examples_plot_text_classification.py`

In order to better understand the role of each file and their content when
uploaded to Hugging Face Hub, refer to this :ref:`user guide <hf_hub>`. You can
refer to :ref:`user guide <model_card>` to see how you can leverage model cards
for documenting your scikit-learn models and enabling reproducibility.

User Guide / API Reference
==========================

.. toctree::
   :maxdepth: 2

   installation
   hf_hub
   model_card
   persistence
   modules/classes
   examples

Community / About
=================
.. toctree::
   :maxdepth: 1

   community
   changes

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
