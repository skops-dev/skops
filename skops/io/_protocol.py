"""The current protocol of the skops version

Notes on updating the protocol:

Every time that a backwards incompatible change to the skops format is made
for the first time within a release, the protocol should be bumped to the next
higher number. The old version of the Node, which knows how to deal with the
old state, should be preserved, registered, and tested. Let's give an example:

- There is a BC breaking change in FunctionNode.
- Since it's the first BC breaking change in the skops format in this release,
  bump skops.io._protocol.PROTOCOL (this file) from version X to X+1.
- Move the old FunctionNode code into 'skops/io/old/_general_vX.py', where 'X'
  is the old protocol.
- Register the _general_vX.FunctionNode in NODE_TYPE_MAPPING inside of
  _persist.py.
- Write a test in test_persist_old.py that shows that the old state can
  still be loaded. Look at test_persist_old.test_function_v0 for inspiration.

Now, if a user loads a FunctionNode state with version X using skops with
version Y>X, the old code will be used instead of the new one. For all other
node types, if there is no loader for version X, skops will automatically use
version Y instead.

"""
PROTOCOL = 2
