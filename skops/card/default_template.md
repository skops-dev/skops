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

## Evaluation Results

You can find the details about evaluation process and the evaluation results.

{{ eval_methods }}

{{ eval_results | default("[More Information Needed]", true)}}

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
