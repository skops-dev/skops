---
{{ card_data }}
---

# Model description

{{ model_description }}

## Intended uses & limitations

{{ limitations }}

## Training Procedure

### Hyperparameters

The model is trained with below hyperparameters.

{{ hyperparameter_table }}

### Model Plot

The model plot is below.

{% autoescape false %}
  {{ model_plot | replace("\n             ", "")}}
{% endautoescape %}
