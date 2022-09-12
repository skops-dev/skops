"""
Easily build interfaces to scikit-learn models using gradio
-----------------------------------------------------------

This guide demonstrates how you can use skops and gradio to build user
interfaces to your models with one line of code, customize and share them.
"""
# %%
# ``gradio`` is a python library that lets you create interfaces on your model.
# It has a class called ``Interface`` that lets you create application
# interfaces to your machine learning models.
# ``gradio`` is integrated with skops, so you can load an interface with only one 
# line of code. You need to initialize an interface, call load method with
# your repository identifier (your user name and name of the model repository)
# prepended with "huggingface/" will load an interface for your model. The interface
# has a dataframe input that takes samples and a dataframe output to return 
# predictions. It also takes the example in the repository that is previously
# pushed with skops. Calling ``launch()`` will launch your application.
import gradio as gr
repo_id = "scikit-learn/tabular-playground"
gr.Interface.load(f"huggingface/{repo_id}")


# You can further customize your UI, add description, title, and more. If you'd 
# like to share your demo, you can set ``share`` to True in ``launch``.
title = "Supersoaker Defective Product Prediction"
description = "This model predicts Supersoaker production line failures. Drag and drop any slice from dataset or edit values as you wish in below dataframe component."
gr.Interface.load(f"huggingface/{repo_id}", title = title, description = description)

# Sharing your local application has time limitations. 
# If you want to share your application continuously, you can deploy it to 
# Hugging Face Spaces.
# For more information, please refer to documentation of ``gradio``.