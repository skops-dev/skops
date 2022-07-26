---
library_name: sklearn
---

# Model description

[More Information Needed]

## Intended uses & limitations

[More Information Needed]

## Training Procedure

### Hyperparameters

The model is trained with below hyperparameters.

<details>
<summary> Click to expand </summary>

| Hyperparameters | Value |
| :-- | :-- |
| copy_X | True |
| fit_intercept | True |
| n_jobs | None |
| normalize | deprecated |
| positive | False |


</details>

### Model Plot

The model plot is below.

<style>#sk-9b2ed01e-2a91-4d1b-b35b-2f95c576d63b {color: black;background-color: white;}#sk-9b2ed01e-2a91-4d1b-b35b-2f95c576d63b pre{padding: 0;}#sk-9b2ed01e-2a91-4d1b-b35b-2f95c576d63b div.sk-toggleable {background-color: white;}#sk-9b2ed01e-2a91-4d1b-b35b-2f95c576d63b label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-9b2ed01e-2a91-4d1b-b35b-2f95c576d63b label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-9b2ed01e-2a91-4d1b-b35b-2f95c576d63b label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-9b2ed01e-2a91-4d1b-b35b-2f95c576d63b div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-9b2ed01e-2a91-4d1b-b35b-2f95c576d63b div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-9b2ed01e-2a91-4d1b-b35b-2f95c576d63b div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-9b2ed01e-2a91-4d1b-b35b-2f95c576d63b input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-9b2ed01e-2a91-4d1b-b35b-2f95c576d63b input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-9b2ed01e-2a91-4d1b-b35b-2f95c576d63b div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-9b2ed01e-2a91-4d1b-b35b-2f95c576d63b div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-9b2ed01e-2a91-4d1b-b35b-2f95c576d63b input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-9b2ed01e-2a91-4d1b-b35b-2f95c576d63b div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-9b2ed01e-2a91-4d1b-b35b-2f95c576d63b div.sk-estimator:hover {background-color: #d4ebff;}#sk-9b2ed01e-2a91-4d1b-b35b-2f95c576d63b div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-9b2ed01e-2a91-4d1b-b35b-2f95c576d63b div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-9b2ed01e-2a91-4d1b-b35b-2f95c576d63b div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 2em;bottom: 0;left: 50%;}#sk-9b2ed01e-2a91-4d1b-b35b-2f95c576d63b div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;}#sk-9b2ed01e-2a91-4d1b-b35b-2f95c576d63b div.sk-item {z-index: 1;}#sk-9b2ed01e-2a91-4d1b-b35b-2f95c576d63b div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;}#sk-9b2ed01e-2a91-4d1b-b35b-2f95c576d63b div.sk-parallel::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 2em;bottom: 0;left: 50%;}#sk-9b2ed01e-2a91-4d1b-b35b-2f95c576d63b div.sk-parallel-item {display: flex;flex-direction: column;position: relative;background-color: white;}#sk-9b2ed01e-2a91-4d1b-b35b-2f95c576d63b div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-9b2ed01e-2a91-4d1b-b35b-2f95c576d63b div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-9b2ed01e-2a91-4d1b-b35b-2f95c576d63b div.sk-parallel-item:only-child::after {width: 0;}#sk-9b2ed01e-2a91-4d1b-b35b-2f95c576d63b div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;position: relative;}#sk-9b2ed01e-2a91-4d1b-b35b-2f95c576d63b div.sk-label label {font-family: monospace;font-weight: bold;background-color: white;display: inline-block;line-height: 1.2em;}#sk-9b2ed01e-2a91-4d1b-b35b-2f95c576d63b div.sk-label-container {position: relative;z-index: 2;text-align: center;}#sk-9b2ed01e-2a91-4d1b-b35b-2f95c576d63b div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-9b2ed01e-2a91-4d1b-b35b-2f95c576d63b div.sk-text-repr-fallback {display: none;}</style><div id="sk-9b2ed01e-2a91-4d1b-b35b-2f95c576d63b" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LinearRegression()</pre><b>Please rerun this cell to show the HTML repr or trust the notebook.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="a4c3291f-bcf3-40f3-b284-246b11aa58ab" type="checkbox" checked><label for="a4c3291f-bcf3-40f3-b284-246b11aa58ab" class="sk-toggleable__label sk-toggleable__label-arrow">LinearRegression</label><div class="sk-toggleable__content"><pre>LinearRegression()</pre></div></div></div></div></div>

# How to Get Started with the Model

Use the code below to get started with the model.

<details>
<summary> Click to expand </summary>

```
[More Information Needed]

```

</details>




# Model Card Authors

This model card is written by following authors:

[More Information Needed]

# Model Card Contact

You can contact the model card authors through following channels:
[More Information Needed]

# Citation

Below you can find information related to citation.

**BibTeX:**
```
[More Information Needed]
```
