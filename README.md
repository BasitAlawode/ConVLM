# ConVLM: Context-Guided Vision-Language Model for Fine-Grained Histopathology Image Classification

![ConVLM](./images/methodolog_5.png)


### Abstract
Vision-Language Models (VLMs) have recently demonstrated exceptional results across various Computational Pathology (CPath) tasks, such as Whole Slide Image (WSI) classification and survival prediction. These models utilize large-scale datasets to align images and text by incorporating language priors during pre-training. However, the separate training of text and vision encoders in current VLMs leads to only coarse-level alignment, failing to capture the fine-level dependencies between image-text pairs. This limitation restricts their generalization in many downstream CPath tasks. In this paper, we propose a novel approach that enhances the capture of finer-level context through language priors, which better represent the fine-grained tissue morphological structures in histology images. We propose a Context-guided Vision-Language Model (ConVLM) that generates contextually relevant visual embeddings from histology images. ConVLM achieves this by employing context-guided token learning and token pruning modules to identify and eliminate contextually irrelevant visual tokens, refining the visual representation. These two modules are integrated into various layers of the ConVLM encoders to progressively learn context-guided visual embeddings, enhancing visual-language interactions. The model is trained end-to-end using a context-guided token learning based loss function. We conducted extensive experiments on 20 histopathology datasets, evaluating both Region of Interest (ROI)-level and cancer subtype WSI-level classification tasks. The results indicate that ConVLM significantly outperforms existing State-of-the-Art (SOTA) vision-language and foundational models

## Environment Setup 

This setup is tested only on Linux.

1. Clone this repository and navigate to ConVLM
```
git clone https://github.com/BasitAlawode/ConVLM.git ConVLM
cd ConVLM
```

2. Install Packages
```
conda create -n convlm python=3.10 -y
conda activate convlm
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

## Text Generation with Quilt-LLaVA

1. open generate_text.py in your favourite text editor

2. Give the path to the images folder by editing line 19.

3. In this work, we have answered the question:
```
questions = ["Can you provide a concise description of the histopathology image shown?"]
```
and the following Quilt-LlaVA configuration has been used:
```
ckpt = "wisdomik/Quilt-Llava-v1.5-7b"
temp, conv_mode = 0, "vicuna_v1"
```
You can change this to the Quilt-LlaVA model you want to use (see lines 22 and 23). 

4. Run:

```
python generate_text.py
```


## Acknowledgement
 - Our work is based on [Quilt-LLaVA](https://github.com/aldraus/quilt-llava) and by extension the [LLaVA model](https://github.com/haotian-liu/LLaVA).

