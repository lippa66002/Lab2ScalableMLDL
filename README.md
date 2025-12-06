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
* **Adjustable Parameters:** Real-time control over Max Tokens, Temperature, Top-p, Min-p, Repetition penalty, and Presence penalty.


## Challenges
This section documents the challenges and approaches taken to improve the inference quality of the fine-tuned small language models (Llama 3.2 1B and Qwen 0.5B).

### Recurring answering loops

During deployment, the models exhibited severe degradation in stop conditions. While they could correctly explain code, they frequently failed to terminate generation. This resulted in "infinite loops" where the model would hallucinate new conversation turns (e.g., generating "User:" prompts) or semantically repeat the same concepts using slightly different phrasing until the token limit was reached.

To resolve this without resource-intensive retraining, we implemented teh following strategies:

1. Architecture-Specific Stop Tokens: We moved away from hardcoded stop tokens and implemented dynamic configuration. We explicitly added <|im_end|>/<|im_start|> for Qwen and <|eot_id|>/<|start_header_id|> for Llama to catch and cut off hallucinated new turns immediately.

2. Min-p Sampling (vs Top-p): We replaced standard nucleus sampling with Min-p (0.05). This sets a dynamic threshold relative to the most likely token, effectively cutting off the low-probability "tail" tokens that often lead small models down repetitive rabbit holes.

3. Dual Penalty System: We combined a multiplicative Repetition Penalty to stop verbatim repetition with an additive Presence Penalty. The presence penalty was critical for stopping the "semantic loops" where the model kept circling back to the same topic despite using different words.

Despite the strategies above we didn't see satisfying improvements in response stability. This limitation could stem from the fundamental capacity of small-parameter models. Models with 0.5B–1B parameters have limited attention heads and "world knowledge," making them prone to mode collapse when they are uncertain. When the model is unsure of the next logical step, it defaults to the most statistically probable path: repeating what it just said.


