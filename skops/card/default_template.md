---
{{ card_data }}
---

# Model description

{{ model_description | default("[More Information Needed]", true)}}

## Intended uses & limitations

{{ limitations | default("[More Information Needed]", true)}}

## Training Procedure

### Hyperparameters

The model is trained with below hyperparameters.

<details>
<summary> Click to expand </summary>

{{ hyperparameter_table }}

</details>

### Model Plot

The model plot is below.

{{ model_plot }}

### Model Examination

## Prediction Reports

Below you can see classification report and confusion matrix.

{{ classification_report | default("[More Information Needed]", true)}}

{{ confusion_matrix | default("[More Information Needed]", true)}}

<img src="./{{ metric_plot }}"/>

## Permutation Importances

{{ permutation_importances | default("[More Information Needed]", true)}}

# How to Get Started with the Model

Use the code below to get started with the model.

<details>
<summary> Click to expand </summary>

```
{{ get_started_code | default("[More Information Needed]", true)}}

```

</details>

# Model Card Authors

This model card is written by following authors:

{{ model_card_authors | default("[More Information Needed]", true)}}

# Model Card Contact

You can contact the model card authors through following channels:
{{ model_card_contact | default("[More Information Needed]", true)}}

# Citation

Below you can find information related to citation.

**BibTeX:**
```
{{ citation_bibtex | default("[More Information Needed]", true)}}
```
