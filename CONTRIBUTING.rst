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

In order to setup ``pre-commit`` hooks, you'd need to run the linter once, ignoring
the outputs:

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

Releases are created using `manual GitHub workflows
<https://docs.github.com/en/actions/managing-workflow-runs/manually-running-a-workflow>`_.
As a maintainer, follow these steps:

1. Check and update the ``docs/changes.rst``
2. For a major release, create a new branch with the name "0.version.X", e.g.
   "0.2.X". This branch will have all tags for all releases under 0.2.
3. Bump the version defined in ``skops/__init__.py``
4. Git grep for any TODO's that need fixing before the release (e.g.
   deprecations). You can do this, for example by:

   .. code:: bash

      git grep -n TODO


5. Create a PR with all the changes and have it reviewed and merged
6. Use the `GitHub action
   <https://github.com/skops-dev/skops/actions/workflows/publish-pypi.yml>`__ to
   create a new release on **TestPyPI**. Check it for correctness `on test.pypi
   <https://test.pypi.org/project/skops/>`_.

7. Create a tag with the format "v0.version", e.g. "v0.2", and push it to the
   remote repository. Use this tag for releasing the package. If there is a
   minor release under the same branch, it would be "v0.2.1" for example.

   .. code:: bash

      git tag v0.2
      git push origin v0.2

8. Use the `GitHub action
   <https://github.com/skops-dev/skops/actions/workflows/publish-pypi.yml>`__ to
   create a new release on **PyPI**. Check it for correctness `pypi
   <https://pypi.org/project/skops/>`_.
9. Create a `new release <https://github.com/skops-dev/skops/releases>`_ on
   GitHub
10. Update the patch version of the package to a new dev version, e.g. from
   ``v0.3.dev0`` to ``v0.4.dev0``
11. Add a section for the new release in the ``docs/changes.rst`` file.
12. Check that the new stable branch of documentation was built correctly on
    `readthedocs <https://readthedocs.org/projects/skops/builds/>`_, and make
    sure all relevant releases are *active*.
