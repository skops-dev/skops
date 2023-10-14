"""Templates for model cards

To add a new template, define it as a dictionary where the key is the section
and the value is the content of the section. If the content is empty but should
be filled by the user, set it to be the ``CONTENT_PLACEHOLDER``.

After defining the template itself, add it as another enum value in the
``Templates`` enum.

Finally, if there is a corresponding section in the new template, some methods
on the ``Card`` class should be adjusted to make use of the template. First of
all, ``_fill_default_sections`` should be used to populate the model card with
the template.

Furthermore, some specific methods might require adjusting. For example, the
``Card._add_hyperparams`` method will add a table of model hyperparameters, but
it needs to know in what section to put them. So if the template contains a
corresponding section, modify the method to put the hyperparameters inside that
section.

"""

from enum import Enum


class Templates(Enum):
    skops = "skops"


CONTENT_PLACEHOLDER = "[More Information Needed]"
"""When there is a section but content has yet to be added by the user, show
this"""

# fmt: off
SKOPS_TEMPLATE = {
    "Model description": CONTENT_PLACEHOLDER,
    "Model description/Intended uses & limitations": CONTENT_PLACEHOLDER,
    "Model description/Training Procedure": CONTENT_PLACEHOLDER,
    "Model description/Training Procedure/Hyperparameters": CONTENT_PLACEHOLDER,
    "Model description/Training Procedure/Model Plot": CONTENT_PLACEHOLDER,
    "Model description/Evaluation Results": CONTENT_PLACEHOLDER,
    "How to Get Started with the Model": CONTENT_PLACEHOLDER,
    "Model Card Authors": (
        f"This model card is written by following authors:\n\n{CONTENT_PLACEHOLDER}"
    ),
    "Model Card Contact": (
        "You can contact the model card authors through following channels:\n"
        f"{CONTENT_PLACEHOLDER}"
    ),
    "Citation": (
        "Below you can find information related to citation.\n\n**BibTeX:**\n```\n"
        f"{CONTENT_PLACEHOLDER}\n```"
    ),
}

# The template below corresponds to the HF Hub default template, but is geared
# towards deep learning models, especially language models, and thus is not a
# good fit for most sklearn models.
_HUB_TEMPLATE = {
    "Model Card": "",
    # Provide a quick summary of what the model is/does.
    "Model Details": "",
    "Model Details/Model Description": "",
    # Provide a longer summary of what this model is.
    "Model Details/Model Description/Developed by": CONTENT_PLACEHOLDER,
    "Model Details/Model Description/Shared by [optional]": CONTENT_PLACEHOLDER,
    "Model Details/Model Description/Model type": CONTENT_PLACEHOLDER,
    "Model Details/Model Description/Language(s) (NLP)": CONTENT_PLACEHOLDER,
    "Model Details/Model Description/License": CONTENT_PLACEHOLDER,
    "Model Details/Model Description/Finetuned from model [optional]":
        CONTENT_PLACEHOLDER,
    "Model Details/Model Description/Resources for more information":
        CONTENT_PLACEHOLDER,

    "Uses": "",
    # Address questions around how the model is intended to be used, including
    # the foreseeable users of the model and those affected by the model.
    "Uses/Direct Use": CONTENT_PLACEHOLDER,
    # This section is for the model use without fine-tuning or plugging into a
    # larger ecosystem/app.
    "Uses/Downstream Use [optional]": CONTENT_PLACEHOLDER,
    # This section is for the model use when fine-tuned for a task, or when
    # plugged into a larger ecosystem/app.
    "Uses/Out-of-Scope Use": CONTENT_PLACEHOLDER,
    # This section addresses misuse, malicious use, and uses that the model will
    # not work well for.

    "Bias, Risks, and Limitations": CONTENT_PLACEHOLDER,
    # This section is meant to convey both technical and sociotechnical
    # limitations.
    "Bias, Risks, and Limitations/Recommendations": (
        "Users (both direct and downstream) should be made aware of the risks, biases "
        "and limitations of the model. More information needed for further "
        "recommendations."
    ),
    # This section is meant to convey recommendations with respect to the bias,
    # risk, and technical limitations.

    "Training Details": "",
    "Training Details/Training Data": CONTENT_PLACEHOLDER,
    # This should link to a Data Card, perhaps with a short stub of information
    # on what the training data is all about as well as documentation related to
    # data pre-processing or additional filtering.
    "Training Details/Training Procedure [optional]": "",
    # This relates heavily to the Technical Specifications. Content here should
    # link to that section when it is relevant to the training procedure.
    "Training Details/Training Procedure [optional]/Preprocessing":
        CONTENT_PLACEHOLDER,
    "Training Details/Training Procedure [optional]/Speeds, Sizes, Times":
        CONTENT_PLACEHOLDER,
    # This section provides information about throughput, start/end time,
    # checkpoint size if relevant, etc.

    "Evaluation": "",
    # This section describes the evaluation protocols and provides the results.
    "Evaluation/Testing Data, Factors & Metrics": "",
    "Evaluation/Testing Data, Factors & Metrics/Testing Data": CONTENT_PLACEHOLDER,
    # This should link to a Data Card if possible
    "Evaluation/Testing Data, Factors & Metrics/Factors": CONTENT_PLACEHOLDER,
    # These are the things the evaluation is disaggregating by, e.g.,
    # subpopulations or domains.
    "Evaluation/Testing Data, Factors & Metrics/Metrics": CONTENT_PLACEHOLDER,
    # These are the evaluation metrics being used, ideally with a description of
    # why.
    "Evaluation/Results": CONTENT_PLACEHOLDER,

    "Model Examination [optional]": CONTENT_PLACEHOLDER,
    # Relevant interpretability work for the model goes here.

    "Environmental Impact": (
        "Carbon emissions can be estimated using the "
        "[Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) "
        "presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700)."
    ),
    # Total emissions (in grams of CO2eq) and additional considerations, such as
    # electricity usage, go here. Edit the suggested text below accordingly"
    "Environmental Impact/Hardware Type": CONTENT_PLACEHOLDER,
    "Environmental Impact/Hours used": CONTENT_PLACEHOLDER,
    "Environmental Impact/Cloud Provider": CONTENT_PLACEHOLDER,
    "Environmental Impact/Compute Region": CONTENT_PLACEHOLDER,
    "Environmental Impact/Carbon Emitted": CONTENT_PLACEHOLDER,

    "Technical Specifications [optional]": "",
    "Technical Specifications [optional]/Model Architecture and Objective":
        CONTENT_PLACEHOLDER,
    "Technical Specifications [optional]/Compute Infrastructure":
        CONTENT_PLACEHOLDER,
    "Technical Specifications [optional]/Compute Infrastructure/Hardware":
        CONTENT_PLACEHOLDER,
    "Technical Specifications [optional]/Compute Infrastructure/Software":
        CONTENT_PLACEHOLDER,

    "Citation [optional]": "",
    # If there is a paper or blog post introducing the model, the APA and Bibtex
    # information for that should go in this section.
    "Citation [optional]/BibTeX": CONTENT_PLACEHOLDER,
    "Citation [optional]/APA": CONTENT_PLACEHOLDER,

    "Glossary [optional]": "",
    # If relevant, include terms and calculations in this section that can help
    # readers understand the model or model card.

    "More Information [optional]": CONTENT_PLACEHOLDER,
    "Model Card Authors [optional]": CONTENT_PLACEHOLDER,
    "Model Card Contact": CONTENT_PLACEHOLDER,
    "How to Get Started with the Model":
        f"""Use the code below to get started with the model.

<details>
<summary> Click to expand </summary>

{CONTENT_PLACEHOLDER}

</details>""",
}
# fmt: on
