---
language: en
thumbnail: "https://camo.githubusercontent.com/7d080b7a769f7fdf64ac0ebeb47b039cb50be35287e3071f9d633f0fe33e7596/68747470733a2f2f692e6962622e636f2f33544331576d472f737065637465722d6c6f676f2d63726f707065642e706e67"
license: apache-2.0
datasets:
- SciDocs
metrics:
- F1
- accuracy
- map
- ndcg
---

## SPECTER

<!-- retrieved on 2022-12-05 | mod: removed trailing whitespaces -->

SPECTER is a pre-trained language model to generate document-level embedding of documents. It is pre-trained on a a powerful signal of document-level relatedness: the citation graph. Unlike existing pretrained language models, SPECTER can be easily applied to downstream applications without task-specific fine-tuning.

Paper: [SPECTER: Document-level Representation Learning using Citation-informed Transformers](https://arxiv.org/pdf/2004.07180.pdf)

Original Repo: [Github](https://github.com/allenai/specter)

Evaluation Benchmark: [SciDocs](https://github.com/allenai/scidocs)

Authors: *Arman Cohan, Sergey Feldman, Iz Beltagy, Doug Downey, Daniel S. Weld*
