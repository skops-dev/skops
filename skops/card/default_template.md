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


fig1
![fig1](fig1.png)



fig2
![fig2](fig2.png)
