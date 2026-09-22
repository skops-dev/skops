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

A release is one run of the `Release workflow
<https://github.com/skops-dev/skops/actions/workflows/publish-pypi.yml>`__.
Given a version, it cuts the release branch, sets the version, builds and
checks the package, publishes it to TestPyPI and then to PyPI, tags the
release, creates the GitHub release, and opens a pull request that moves
``main`` on to the next development version. As a maintainer, follow these
steps:

1. Make sure ``docs/changes.rst`` has a complete section for the release. The
   section is named like the tag: ``v0.16`` for version ``0.16.0`` and
   ``v0.16.1`` for a bug fix release ``0.16.1``. The workflow refuses to release
   a version whose section is missing or empty.
2. Git grep for any TODO's that need fixing before the release (e.g.
   deprecations):

   .. code:: bash

      git grep -n TODO

3. Start the `Release workflow
   <https://github.com/skops-dev/skops/actions/workflows/publish-pypi.yml>`__
   from the ``main`` branch and enter the version to release, e.g. ``0.16.0``.
   This cuts the ``0.16.X`` branch from ``main``. For a bug fix release, e.g.
   ``0.16.1``, the ``0.16.X`` branch must already contain the backported fixes;
   the workflow then releases from that branch and leaves ``main`` alone.
4. The workflow builds and checks the package, then pauses for a maintainer to
   approve the ``publish-pypi`` environment before publishing to **TestPyPI**.
   Check the release `on TestPyPI <https://test.pypi.org/project/skops/>`_, then
   approve the environment a second time to publish to **PyPI**. After that
   the workflow pushes the tag, creates the `GitHub release
   <https://github.com/skops-dev/skops/releases>`_ with generated notes, and,
   for a new minor release, opens a pull request against ``main`` that bumps
   the version to the next development version and adds an empty changelog
   section for it.
5. Merge that pull request, and the pull request that the conda-forge bot opens
   on the `feedstock <https://github.com/conda-forge/skops-feedstock>`_. If any
   dependency versions changed, make sure they are reflected in the feedstock
   recipe.
6. Check that the documentation for the new version was built correctly on
   `readthedocs <https://readthedocs.org/projects/skops/builds/>`_, and make
   sure all relevant releases are *active*.

The ``dry_run`` option of the workflow runs everything up to and including the
build without pushing, publishing, tagging or opening a pull request, which is
useful to try changes to the workflow, also on a fork. Should the workflow fail
after the package was published, the remaining steps are small enough to do by
hand: the tag points at the ``REL set version to ...`` commit on the release
branch, and the version bump on ``main`` is ``python scripts/release.py
start-dev 0.17``.
