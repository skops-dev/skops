import streamlit as st


def add_back_button(key):
    def fn():
        st.session_state.screen.state = "start"

    st.button("Back", help="Get back to the start screen", on_click=fn, key=key)


help_md = """# Create a Hugging Face model repository for scikit learn models

This page aims to provide a simple interface to use the
[`skops`](https://skops.readthedocs.io/) model card and HF Hub creation
utilities.

Below, we will explain the steps involved to create your own model repository to
host your scikit-learn model:

1. Prepare the model repository
2. Edit the model card
3. Create the model repository on Hugging Face Hub

## Step 1: Prepare the model repository

In this step, you do the necessary preparation work to create a [model
repository on Hugging Face Hub](https://huggingface.co/docs/hub/models).

### Upload a model

Here you should upload the sklearn model we want to present in the model
repository. The model should be stored either as a ``pickle`` file or it should
use the [secure skops persistence
format](https://skops.readthedocs.io/en/stable/persistence.html). Later, this
model will be uploaded to the model repository so that you can share it with
others.

The uploaded model should be a scikit-learn model or a model that is compatible
with the sklearn API, e.g. using [XGBoost sklearn
wrapper](https://xgboost.readthedocs.io/en/stable/python/python_api.html#module-xgboost.sklearn)
when it's an XGBoost model.

If you just want to test out the application and don't want to upload a model, a
dummy model will be used instead.

### Upload input data

It's possible to upload input data as a csv file. If that is done, the first few
rows of the input data will be used as sample data for the model, e.g. when
trying out the [inference API](https://huggingface.co/inference-api).


### Choose the task type

Choose the type of task that the model is intended to solve. It can be either
classification or regression, with input data being either tabular in nature or
text.

### Requirements

This is the list of Python requirements needed to run the model.

### Choose the model card template

This is the final step and choosing one of the options will bring you to the
editing step.

#### Create a new skops model card

This is the recommended way of getting started. The skops model card template
prefills the model card with some [useful
contents](https://skops.readthedocs.io/en/stable/model_card.html#model-card-content)
that you probably want to have in most model cards. Don't worry: If you don't
like part of the content, you can always edit or delete it later.

#### Create a new empty model card

If you want to start the model card completely from scratch, that's also
possible by choosing this option. It will generate a completely empty model card
for you that you can fashion to your liking.

#### Load existing model card from HF Hub

If you want to use an existing model card and edit it, that's also possible.
Please enter the Hugging Face Hub repository ID here and the corresponding model
card will be loaded. The repo ID is typically someting like `username/reponame`,
e.g. `openai/whisper-small`. Some models also omit the user name, e.g. `gpt2`.

Note that when you choose an existing model card, a couple of files will be
downloaded, because they may be required to render the model card (e.g. images).
Therefore, depending on the repository, this step may take a bit.

If you notice any problems when rendering the existing model card, please let us
know by [creating an issue](https://github.com/skops-dev/skops/issues).

## Step 2: Edit the model card

Before creating the model repository, it is crucial to ensure that the [model
card](https://huggingface.co/blog/model-cards) is edited to best represent the
model you're working on. This can be achieved in the editing step, which is
described in more detail below.

### Editing sidbar

In the left sidebar, you will be able to edit the model card, whereas the main
screen is reserved for rendering the model card so that you see what you will
get. We will start by describing the editing sidebar.

Tip: You should increase the width of the side bar if it is too narrow for your
taste.

#### Undo, redo & reset

On top of the side bar, you have the option to undo, redo, and reset the last
operation you did. Say, you accidentally made a change, just press the `Undo`
button to undo this change. Similarly, if you want to undo your undo operation,
press the `Redo` button. Finally, if you press `Reset`, all your operations will
be undone (but don't worry if you click the button accidentally, you can redo
all of them if you want).

#### Save, create repo & delete

These buttons are intended for when you finished editing the model card. When
you click on `Save`, you will get the option to download the model card as a
markdown file.

When clicking the `Create Repo` button, you will be taken to the next screen,
which offers you to create a model repository on Hugging Face Hub. This step
will be explained in more detail further below.

Finally, you can click on `Delete` to completely discard all the changes you
made and be taken back to the start screen of the app. Be careful, any change
you made will be lost. It is thus advised to first save the model card before
pressing `Delete`.

#### Edit a section

Each section has its own form field, which allows you to make edits. Change the
name of the section or change the content (or both), then click `Update` to see
a preview of your change. As with all other operations, you can undo the change
by clicking on `Undo`.

#### Delete a section

Below the form field for editing the section, you will find a `Delete` button
(including the name of the section to make it clear which section it refers to).
If you click that button, the whole section, _including its subsections_, will
be deleted. Again, click on `Undo` if you accidentally deleted something that
you want to keep.

#### Add section below

If you click on this button, a new subsection wil be created under the current
section. This will create a section with a dummy title and dummy content, which
you can then edit.

Note that this will create a new _subsection_. If there are already existing
subsections in the current section, the new subsection will be created _below_
those existing subsections. So the new subsection you create might not appear
exactly where you expect it to appear. To illustrate this, assume that we have
the following sections and subsections:

- Section A
  - Subsection A.1
  - Subsection A.2
- Section B

If you create a new section below "Section A", it will be created on the same
level, and below of, "Subsection A.2", resulting in the following structure:

- Section A
  - Subsection A.1
  - Subsection A.2
  - NEW SUBSECTION
- Section B

If you create a new section below the "Subsection A.1", you will actually create
a sub-subsection, resulting in the following structure instead:

- Section A
  - Subsection A.1
    - NEW SUB-SUBSECTION
  - Subsection A.2
- Section B

Hopefully, this clarifies things. Unfortunately, there is no possibility (yet)
to re-order sections.

#### Add figure below

This button works quite similarly to adding a new section. The main difference
is that instead of having a text area to enter content, you will be asked to
upload an image file. By default, a dummy image will be shown in the preview.

#### Add metrics (only skops template)

If you have chosen the skops template, you will see an additional field called
`Add metrics` near the top of the side bar. Here you can choose metrics you want
to be shown in the model card, e.g. the accuracy the model achieved on a
specific dataset. Please enter one metric per line, with the metric name on the
left, then an `=` sign, and the value on the right, e.g. `accuracy on test set =
0.85`.

After pressing `update`, the metrics will be shown in a table in the section
`Model description/Evaluation Results`. You can always add or remove metrics
from this field later. If you want to delete this section completely, look for
its edit form further below and press `Delete`. There, you can also edit the
table in a more fine grained way, e.g. by changing the alignment.

If you don't use the skops template and still want to add a table, it is
possible to do that, but it's requires a bit more work. Add a new section as
described above, then, in the text area, create a table using the [markdown
table
syntax](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/organizing-information-with-tables#creating-a-table).

### Model card visualization

The main part of the page will show you what the final model card will look
like.

#### Metadata

On the very top, you can see the metadata of the model card (it is collapsed by
default). The metadata can be very useful for features on the HF Hub, e.g.
allowing other users to find your model by a given tag.

Right now, it is not possible to edit the metadata directly from here. But don't
worry, once you have created the model card repository, you can easily edit the
metadata there.

#### Table of Contents

For your convenience, a table of contents is also shown at the top (collapsed by
default). This is useful if you have a bigger model card and want to see the
overview of all its contents.

#### Markdown preview

Finally, the model card itself is shown. This is how the model card will look
like once it is saved as markdown and then rendered.

## Step 3: Creating a model repository

After you have finished editing the model card, it is time to create a model
repository on Hugging Face Hub. Click on `Create Repo` and you will be taken to
the final step of the process.

### Back & Delete

If you find yourself wanting to make more edits to the model card, just click on
the `Back` button and you'll be brought back to the editing step.

You can also click `Delete`, which will discard all your changes and bring you
back to the start page. Be careful: This step cannot be undone and all your
progress will be lost.

### Files included in the repository

For your convenience, this will show a preview of all the files included in the
repository, as well as their sizes. Don't create a repository if you see files
there that you don't want to be uploaded.

### Privacy settings

By default, a private repository will be created. If you untick this box, it
will be public instead. More information on what that implies can be found in
the [docs on repository
settings](https://huggingface.co/docs/hub/repositories-settings).

### Name of the repository

Here you have to enter the name of the repository. Typically, that's something
like `username/reponame` or `organame/reponame`. This field is mandatory and you
should ensure that the corresponding repository ID does not exist yet.

### Enter your Hugging Face token

Here you need to paste your Hugging Face token, which is used for
authentication. The token can be found [here](https://hf.co/settings/token) and
it always starts with "hf_". Entering a token is necessary to create a
repository.

Note that if you don't already have an account on Hugging Face, you need to
create one to get a token. It's free.

### Create a new repository

Once all the required fields are filled, click on this button to create the
repository. Depending on the size, it may take a couple of seconds to finish.
Once it is created, you will see a success notification that includes the link
to the repository. Congratulations, you're done!

## Troubleshooting

### Not all skops features available

This app is based on the [skops model card
feature](https://skops.readthedocs.io/en/stable/model_card.html#model-card-content).
However, it does not support all the options that are available there. If you
want to use all those options in a programmatic fashion, please follow the link
and read up on what it takes to create a model card with skops. The full power
of the `Card` class is documented
[here](https://skops.readthedocs.io/en/stable/modules/classes.html#skops.card.Card).

### Strange behavior

If the app behaves strangely, shows error messages, or renders incorrectly, it
may be necessary to refresh the browser tab. This will take you back to the
start page, with all progress being lost. If you can reproduce that behavior,
please [creating an issue](https://github.com/skops-dev/skops/issues) and let us
know.

### Contact

If you want to contact us, you can join our discord channel. To do that, follow
[these
instructions](https://skops.readthedocs.io/en/stable/community.html#discord).
"""


def add_help_content():
    # This is the exact same text as in the README.md of this space
    st.markdown(help_md)


def help_page():
    add_back_button(key="help_get_back")
    add_help_content()
    add_back_button(key="help_get_back2")  # names must be unique
