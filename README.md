# :sparkles: ScienceChatbot
ScienceChatbot contains a full-stack educational visual reasoning chatbot based on Qwen3-VL. We fine-tune models on the ScienceQA dataset for multimodal question answering and explanation generation, and provide a front-end and back-end pipeline for image-based student question solving.

## :rocket: About
ScienceChatbot presents an educational multimodal large language model system designed for multimodal question answering and reasoning in science-learning scenarios for students in elementary school and high school. The system is fine-tuned on the ScienceQA dataset and supports image-based problem input, answer prediction, and explanation generation.

The goal of this project is to bridge vision language model (VLM) with real-world educational applications, enabling students to upload question images (e.g., science diagrams, charts) and receive both the correct answer and detailed explanations.

The repository includes:

* Model fine-tuning and evaluation code based on ScienceQA
* Multimodal inference pipeline
* Frontend interface and backend service 

This work focuses on multimodal reasoning, answer prediction, and explanation generation for science visual question answering tasks.

## 📖 Dataset: ScienceQA
This project is built upon the [ScienceQA](https://github.com/lupantech/ScienceQA) dataset, a multimodal multiple-choice question answering benchmark designed for elementary and high school science education.

According to the original [ScienceQA](https://lupantech.github.io/papers/neurips22_scienceqa.pdf) paper, the dataset contains over 20,000 science questions covering diverse subjects such as natural science, social science, and language science. Each question may include:

* A question stem
* Multiple answer choices
* Optional image context (e.g., diagrams, illustrations, charts)
* Supporting textual context
* A human-written rationale explaining the correct answer

ScienceQA is specifically designed to evaluate multimodal reasoning, requiring models to integrate visual information with textual context and domain knowledge to produce accurate answers.

In this project, ScienceQA serves as:

* The primary benchmark for fine-tuning the vision-language model
* A supervised source for answer prediction and explanation generation
* A testbed for studying image-grounded educational reasoning

By leveraging ScienceQA, the system aims to bridge multimodal large language models with real-world educational applications.

## 🔭 Model & Multimodal Reasoning

### 🔬 Model: Qwen3-VL-8B-Instruct

This project is built upon [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct), an instruction-tuned large-scale vision-language model developed by Alibaba Cloud. The model integrates visual encoding and large language modeling capabilities, enabling joint reasoning over image and text inputs.

* Image-text understanding

* Multimodal reasoning

* Visual question answering

* Step-by-step explanation generation

The model accepts interleaved image and text inputs and generates natural language outputs conditioned on both modalities.

### 📱 Model Input 

The model follows an interleaved multimodal message structure:

```
messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": system_prompt}],
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "path/to/image.png"},
            {"type": "text", "text": question_and_options},
        ],
    },
]
```

The image is provided first, followed by the textual question and answer options.
This allows the model to perform image-grounded reasoning over the provided visual context.

### 🖼️ Model Output

The model is prompted to generate:

* A single correct answer (e.g., A/B/C/D)

* A structured explanation (rationale) supporting the prediction

This aligns with the ScienceQA setting, which includes human-written rationales.
