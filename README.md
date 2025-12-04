---
title: Iris
emoji: 💬
colorFrom: yellow
colorTo: purple
sdk: gradio
sdk_version: 5.42.0
app_file: app.py
pinned: false
hf_oauth: true
hf_oauth_scopes:
- inference-api
license: apache-2.0
---

# ID2223 Lab 2: Fine-Tuning LLMs for Code Assistance

This project is part of the ID2223 Scalable Machine Learning and Deep Learning course at KTH (HT2025). The goal is to demonstrate the fine-tuning of Large Language Models (LLMs) on specific datasets and deploying them via a serverless UI.

## Project Overview

We have fine-tuned efficient, small-scale LLMs to act as specialized coding assistants. The application allows users to switch between different model architectures and training datasets to compare performance on coding tasks.

**Deployed App:** [https://huggingface.co/spaces/rylla/Iris](https://huggingface.co/spaces/rylla/Iris)

### Features

* **Multi-Model Support:** Switch instantly between **Llama 3.2 1B** and **Qwen 2.5 0.5B**.
* **Specialized Datasets:**
    * *Finetome:* Optimized for general instruction following.
    * *Code Docs:* Specialized training on code documentation and technical content.
* **Code Context Window:** A dedicated split-view editor allowing users to paste code snippets that the LLM uses as context for its answers.
* **GGUF Optimization:** Models are quantized (Q4_K_M) and loaded via `llama.cpp` to ensure fast inference on CPU-based environments like Hugging Face Spaces.
* **Adjustable Parameters:** Real-time control over Temperature, Top-p, and Max Tokens.