---
dataset_info:
  features:
  - name: image
    dtype: image
  - name: category1
    dtype: string
  - name: category2
    dtype: string
  - name: category3
    dtype: string
  - name: text
    dtype: string
  - name: item_ID
    dtype: string
  splits:
  - name: data
    num_bytes: 4235530431.72
    num_examples: 201624
  download_size: 3466991670
  dataset_size: 4235530431.72
configs:
- config_name: default
  data_files:
  - split: data
    path: data/data-*
license: apache-2.0
---
**Disclaimer**: We do not own this dataset. Fashion200K dataset is a public dataset which can be accessed through its [Github page](https://github.com/xthan/fashion-200k?tab=readme-ov-file).

This dataset was used to evaluate Marqo-FashionCLIP and Marqo-FashionSigLIP - see details below.

# Marqo-FashionSigLIP Model Card
Marqo-FashionSigLIP leverages Generalised Contrastive Learning ([GCL](https://www.marqo.ai/blog/generalized-contrastive-learning-for-multi-modal-retrieval-and-ranking)) which allows the model to be trained on not just text descriptions but also categories, style, colors, materials, keywords and fine-details to provide highly relevant search results on fashion products. 
The model was fine-tuned from ViT-B-16-SigLIP (webli). 

**Github Page**: [Marqo-FashionCLIP](https://github.com/marqo-ai/marqo-FashionCLIP)

**Blog**: [Marqo Blog](https://www.marqo.ai/blog/search-model-for-fashion)


## Usage
The model can be seamlessly used with [OpenCLIP](https://github.com/mlfoundations/open_clip) by

```python
import open_clip
model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:Marqo/marqo-fashionSigLIP')
tokenizer = open_clip.get_tokenizer('hf-hub:Marqo/marqo-fashionSigLIP')
import torch
from PIL import Image
image = preprocess_val(Image.open("docs/fashion-hippo.png")).unsqueeze(0)
text = tokenizer(["a hat", "a t-shirt", "shoes"])
with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
print("Label probs:", text_probs)
```

## Benchmark Results
Average evaluation results on 6 public multimodal fashion datasets ([Atlas](https://huggingface.co/datasets/Marqo/atlas), [DeepFashion (In-shop)](https://huggingface.co/datasets/Marqo/deepfashion-inshop), [DeepFashion (Multimodal)](https://huggingface.co/datasets/Marqo/deepfashion-multimodal), [Fashion200k](https://huggingface.co/datasets/Marqo/fashion200k), [KAGL](https://huggingface.co/datasets/Marqo/KAGL), and [Polyvore](https://huggingface.co/datasets/Marqo/polyvore)) are reported below: 

**Text-To-Image (Averaged across 6 datasets)**
| Model                      | AvgRecall   | Recall@1   | Recall@10   | MRR       |
|----------------------------|-------------|------------|-------------|-----------|
| Marqo-FashionSigLIP        | **0.231**   | **0.121**  | **0.340**   | **0.239** |
| FashionCLIP2.0             | 0.163       | 0.077      | 0.249       | 0.165     |
| OpenFashionCLIP            | 0.132       | 0.060      | 0.204       | 0.135     |
| ViT-B-16-laion2b_s34b_b88k | 0.174       | 0.088      | 0.261       | 0.180     |
| ViT-B-16-SigLIP-webli      | 0.212       | 0.111      | 0.314       | 0.214     |

**Category-To-Product (Averaged across 5 datasets)**
| Model                      | AvgP      | P@1       | P@10      | MRR       |
|----------------------------|-----------|-----------|-----------|-----------|
| Marqo-FashionSigLIP        | **0.737** | **0.758** | **0.716** | **0.812** |
| FashionCLIP2.0             | 0.684     | 0.681     | 0.686     | 0.741     |
| OpenFashionCLIP            | 0.646     | 0.653     | 0.639     | 0.720     |
| ViT-B-16-laion2b_s34b_b88k | 0.662     | 0.673     | 0.652     | 0.743     |
| ViT-B-16-SigLIP-webli      | 0.688     | 0.690     | 0.685     | 0.751     |

**Sub-Category-To-Product (Averaged across 4 datasets)**
| Model                      | AvgP      | P@1       | P@10      | MRR       |
|----------------------------|-----------|-----------|-----------|-----------|
| Marqo-FashionSigLIP        | **0.725** | **0.767** | **0.683** | **0.811** |
| FashionCLIP2.0             | 0.657     | 0.676     | 0.638     | 0.733     |
| OpenFashionCLIP            | 0.598     | 0.619     | 0.578     | 0.689     |
| ViT-B-16-laion2b_s34b_b88k | 0.638     | 0.651     | 0.624     | 0.712     |
| ViT-B-16-SigLIP-webli      | 0.643     | 0.643     | 0.643     | 0.726     |


When using the datset, cite the original work.
```
@inproceedings{han2017automatic,
  title = {Automatic Spatially-aware Fashion Concept Discovery},
  author = {Han, Xintong and Wu, Zuxuan and Huang, Phoenix X. and Zhang, Xiao and Zhu, Menglong and Li, Yuan and Zhao, Yang  and Davis, Larry S.},
  booktitle = {ICCV},
  year  = {2017},
}
```