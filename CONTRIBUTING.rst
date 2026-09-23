Contributing to skops
=====================

Please follow this workflow when contributing to skops:

- Fork the repository under your own user
- Clone the repository locally
- Create a new branch for your changes
- Add your changes to the branch
- Commit your changes
- Push your branch to the remote repository
- Create a pull request on GitHub

Issue Titles / Commit Messages
------------------------------

When creating a pull request, please use a descriptive title. You can prefix
the title to indicate the type of it:

- ``DOC``: documentation changes
- ``FEAT/FEA``: new major features
- ``ENH``: enhancements to existing features with user facing implications
- ``CI``: continuous integration, sometimes overlaps with MNT
- ``MNT/MAINT``: maintenance, technical debt, etc
- ``FIX``: bug fixes
- ``TST``: new tests, refactoring tests
- ``PERF``: performance improvements

If a contributor forgets to prefix the title, a maintainer can add the prefix
when merging into ``main``. While merging, it is recommended that the
maintainer refines the commit message to add a short description of what the PR
being merged does.

Review Process
--------------

Don't hesitate to ping @skops-dev/maintainers in your issues and pull requests
if you don't receive a review in a timely manner. We try to review all pull
requests as soon as we can.

If you have permissions, you should almost never merge your own pull request
unless it's a hotfix and needs to be merged really quick and it's not a major
change.

Otherwise pull requests can be merged if at least one other person has approved
it on GitHub. Please don't merge them until all outstanding comments are
addressed or the discussions are concluded and people have agreed to tackle
them in future pull requests.

Working on Existing Issues
--------------------------

If you intend to work on an issue, leave a comment and state your intentions.
Also feel free to ask for clarifications if you're not sure what the issue
entails. If you don't understand an issue, it's on us, not on you!

Setting up the dev environment
------------------------------

We use `pixi <https://github.com/prefix-dev/pixi>`_ in our CI and development
workflows and recommend you use it to test the changes you make.

Once you have ``pixi`` installed, you can run the tests with:

.. code:: bash

   pixi run tests

And you can choose an environment to run the tests with:

.. code:: bash

   pixi run -e ci-sklearn15 tests

We use `ruff <https://docs.astral.sh/ruff/>`_ for formatting and linting, and
`pyrefly <https://pyrefly.org/>`_ for type checking. Both are run through
``pre-commit``, with their versions pinned in ``.pre-commit-config.yaml``. In order
to setup the ``pre-commit`` hooks, you'd need to run the linter once, ignoring the
outputs:

.. code:: bash

   pixi run -e lint lint

VSCode-like IDEs automatically detect ``pixi`` environments and you can use them as
your python interpreter.

Running Tests Manually
~~~~~~~~~~~~~~~~~~~~~~

You can get an interactive shell into an environment with the nightly build of
scikit-learn and all other required dependencies with:

.. code:: bash

   pixi shell

``skops`` uses ``pytest`` as its test runner, just run it from the project root:

.. code:: bash

   pytest


Releases
========

Releases are cut from ``main``; there are no bug fix releases of older
versions. Pushing a tag such as ``v0.16.0`` starts the `Release workflow
<https://github.com/skops-dev/skops/actions/workflows/publish-pypi.yml>`__,
which builds the package, checks that the tag matches the version in
``skops/__init__.py``, publishes to TestPyPI and then to PyPI, and creates the
GitHub release with generated notes. Publishing waits for a maintainer to
approve the ``publish-pypi`` environment, once for TestPyPI and once for PyPI.
As a maintainer, follow these steps:

1. Check that ``docs/changes.rst`` has a complete section for the release, e.g.
   ``v0.16``, and git grep for any TODO's that need fixing before the release
   (e.g. deprecations):

   .. code:: bash

      git grep -n TODO

2. Open a pull request that sets ``__version__`` in ``skops/__init__.py`` to
   the release version, e.g. ``0.16.0``, and merge it.
3. Tag the merge commit with ``v`` followed by the version and push the tag to
   ``skops-dev/skops`` (the ``upstream`` remote here):

   .. code:: bash

      git fetch upstream
      git tag v0.16.0 upstream/main
      git push upstream v0.16.0

4. In the workflow run, approve the ``publish-pypi`` environment to publish to
   **TestPyPI**. Check the release `on TestPyPI
   <https://test.pypi.org/project/skops/>`_, then approve again to publish to
   **PyPI**. The workflow then creates the `GitHub release
   <https://github.com/skops-dev/skops/releases>`_.
5. Open a pull request that sets ``__version__`` to the next development
   version, e.g. ``0.17.dev0``, and adds an empty ``v0.17`` section at the top
   of ``docs/changes.rst``, and merge it.
6. Merge the pull request that the conda-forge bot opens on the `feedstock
   <https://github.com/conda-forge/skops-feedstock>`_. If any dependency
   versions changed, make sure they are reflected in the feedstock recipe.
7. Check that the documentation for the new version was built correctly on
   `readthedocs <https://readthedocs.org/projects/skops/builds/>`_, and make
   sure all relevant releases are *active*.

To try changes to the workflow without publishing anything, start it by hand
from the "Actions" tab: without a tag it only builds and checks the package.
