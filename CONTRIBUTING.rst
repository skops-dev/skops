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

Following these steps you can prepare a dev environment for yourself to
contribute to `skops`.

Using conda/mamba
~~~~~~~~~~~~~~~~~

.. code:: bash

          mamba create -c conda-forge -n skops python=3.10
          mamba activate skops
          python -m pip install -e ".[tests,docs]"
          # add pre-commit hooks
          mamba install -c conda-forge pre-commit
          pre-commit install

You can also replace the above `mamba` commands with `conda` if you don't have
`mamba` installed.


Running Tests
~~~~~~~~~~~~~

skops uses pytest as its test runner, just run it from the project root:

.. code:: bash

   pytest

Certain tests require internet access to run, and they typically take slightly
longer to run than other tests. If you'd like to skip those tests, you can add
``-m not network`` to your ``pytest`` command, or ``-m network`` to only run
those tests. For example, you can run all tests except the ones requiring
internet with:

.. code:: bash

   pytest -m "not network"


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
   deprecations)
5. Create a PR with all the changes and have it reviewed and merged
6. Create a tag with the format "v0.version", e.g. "v0.2", and push it to the
   remote repository. Use this tag for releasing the package. If there is a
   minor release under the same branch, it would be "v0.2.1" for example.
6. Use the `GitHub action
   <https://github.com/skops-dev/skops/actions/workflows/publish-pypi.yml>`__ to
   create a new release on **TestPyPI**. Check it for correctness `on test.pypi
   <https://test.pypi.org/project/skops/>`_.
7. Use the `GitHub action
   <https://github.com/skops-dev/skops/actions/workflows/publish-pypi.yml>`__ to
   create a new release on **PyPI**. Check it for correctness `pypi
   <https://pypi.org/project/skops/>`_.
8. Create a `new release <https://github.com/skops-dev/skops/releases>`_ on
   GitHub
9. Update the patch version of the package to a new dev version, e.g. from
   ``v0.3.0`` to ``v0.3.dev1``
10. Check that the new stable branch of documentation was built correctly on
    `readthedocs <https://readthedocs.org/projects/skops/builds/>`_, and make
    sure all relevant releases are *active*.
