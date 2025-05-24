# 🧠 GenNarrate: Script & Dialogue Generator

GenNarrate is a local generative storytelling app powered by open-source LLMs. It enables users to generate creative scripts and refine dialogue using GPT-Neo and BART models. The app includes built-in evaluation metrics like BLEU and Perplexity for assessing output quality — all within a responsive Streamlit interface.

https://gennarrate-hum8vy9gxs6pnqexsorqmh.streamlit.app/

## ✨ Features

- 🎬 **Script Generation** using GPT-Neo
- 🎭 **Dialogue Refinement** using BART
- 📊 **Evaluation** with BLEU and Perplexity scores
- 💻 **Offline Inference** (no API keys needed)
- 🧪 **Prompt Selector**, Typing Animation, and Output Downloads

## 🛠 Tech Stack

- Python, Streamlit
- Hugging Face Transformers (`GPT-Neo`, `facebook/bart-large-cnn`)
- NLTK (for BLEU scoring)
- PyTorch (for model inference)

## 🚀 How to Run

```bash
# Install dependencies
pip install streamlit torch transformers nltk

# Run the app
streamlit run main.py

