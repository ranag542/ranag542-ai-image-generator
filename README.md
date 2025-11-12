# 🖼️ Image Generator (Stable Diffusion CLI)

Generate multiple images from a text prompt using **Stable Diffusion** powered by the [🤗 Diffusers](https://github.com/huggingface/diffusers) library.

---

## 🚀 Features
- Generate high-quality images from any text prompt  
- Support for CUDA GPUs (automatic detection)  
- Simple command-line interface  
- Saves all generated images to an output folder  

---

## 🧠 Requirements
Make sure you have **Python 3.8+** and a **modern NVIDIA GPU** (for best performance).

Install dependencies:
```bash
pip install diffusers transformers torch accelerate safetensors
