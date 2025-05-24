import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
import time

# Must be first Streamlit command
st.set_page_config(page_title="GenNarrate", layout="wide")

# Download NLTK tokenizer data
nltk.download('punkt')

# Model names
GPT_MODEL = "EleutherAI/gpt-neo-125M"
REFINER_MODEL = "facebook/bart-large-cnn"

# Caching for heavy model loading
@st.cache_resource
def load_gpt_model():
    tokenizer = AutoTokenizer.from_pretrained(GPT_MODEL)
    model = AutoModelForCausalLM.from_pretrained(GPT_MODEL)
    return tokenizer, model

@st.cache_resource
def load_refiner():
    return pipeline("text2text-generation", model=REFINER_MODEL)

tokenizer, model = load_gpt_model()
refiner = load_refiner()

# Utility Functions
def generate_script(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(
        **inputs,
        max_length=400,
        temperature=0.7,
        top_k=50,
        repetition_penalty=1.2,
        do_sample=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def refine_dialogue(dialogue):
    refined = refiner(dialogue, max_length=50, do_sample=False)
    return refined[0]['generated_text']

def remove_repetition(text):
    sentences = text.split(". ")
    unique_sentences = []
    for sentence in sentences:
        if sentence not in unique_sentences:
            unique_sentences.append(sentence)
    return ". ".join(unique_sentences)

def calculate_perplexity(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        return torch.exp(loss).item()

def compute_bleu(reference, candidate):
    ref_tokens = nltk.word_tokenize(reference)
    cand_tokens = nltk.word_tokenize(candidate)
    smoothie = SmoothingFunction().method4
    return sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smoothie)

def animated_typing(text):
    for line in text.split("\n"):
        st.markdown(f"<div style='font-family:monospace; color:#e8e8e8;'>{line}</div>", unsafe_allow_html=True)
        time.sleep(0.02)

# 🎯 Example Prompt Options
example_prompts = {
    "Cyberpunk Scene": "A futuristic Tokyo alleyway at night, glowing with neon signs and light rain.",
    "Fantasy Kingdom": "A majestic castle floating above the clouds, with dragons circling in the distance.",
    "Sci-Fi Battle": "A massive space battle between alien ships and human fighters in a purple galaxy.",
    "Nature Landscape": "A peaceful forest with golden sunlight filtering through ancient trees and a glowing waterfall.",
    "Noir Detective": "A 1940s detective's office at night, dimly lit by a desk lamp and filled with cigarette smoke.",
    "Romantic Dinner": "A candlelit rooftop restaurant overlooking a bustling city, with soft music playing in the background.",
    "Action Scene": "A high-speed car chase through a futuristic city, with explosions and flying drones."
}

# UI Header
st.markdown("<h1 style='text-align: center;'>🧠 GenNarrate: Script & Dialogue Generator</h1>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center; font-size:16px; color:gray;'>Use AI to generate creative scripts and refine dialogue interactively.</div>", unsafe_allow_html=True)

# Tabs: Script Generator | Dialogue Refiner
tabs = st.tabs(["📜 Script Generator", "🎭 Dialogue Refiner"])

# --- Script Generator ---
with tabs[0]:
    st.markdown("<h3 style='color:#4A90E2;'>📜 Generate Script</h3>", unsafe_allow_html=True)
    selected_example = st.selectbox("Choose an example prompt:", ["None"] + list(example_prompts.keys()))
    default_prompt = example_prompts[selected_example] if selected_example != "None" else ""

    script_prompt = st.text_input("Enter your script idea:", value=default_prompt)
    if st.button("Generate Script"):
        if script_prompt.strip() == "":
            st.warning("Please enter a script idea.")
        else:
            with st.spinner("Generating script..."):
                try:
                    script = generate_script(script_prompt)
                    script = remove_repetition(script)
                    st.session_state.generated_script = script
                    st.session_state.script_ppl = calculate_perplexity(script)
                except Exception as e:
                    st.error(f"Script generation failed: {e}")

    if "generated_script" in st.session_state:
        st.subheader("📝 Generated Script")
        animated_typing(st.session_state.generated_script)

        st.subheader("📏 Perplexity Score")
        st.write(f"{st.session_state.script_ppl:.2f}")

        st.download_button("📥 Download Script", st.session_state.generated_script, file_name="generated_script.txt")

# --- Dialogue Refiner ---
with tabs[1]:
    st.markdown("<h3 style='color:#50E3C2;'>🎭 Refine Dialogue</h3>", unsafe_allow_html=True)
    raw_dialogue = st.text_input("Enter raw dialogue:")
    reference_dialogue = st.text_input("Optional: Provide reference dialogue for evaluation:")

    if st.button("Refine Dialogue"):
        if raw_dialogue.strip() == "":
            st.warning("Please enter dialogue to refine.")
        else:
            with st.spinner("Refining..."):
                try:
                    improved = refine_dialogue(raw_dialogue)
                    st.session_state.refined_dialogue = improved

                    if reference_dialogue.strip():
                        bleu = compute_bleu(reference_dialogue, improved)
                        st.session_state.bleu_score = bleu
                except Exception as e:
                    st.error(f"Refinement failed: {e}")

    if "refined_dialogue" in st.session_state:
        st.subheader("✨ Refined Dialogue")
        animated_typing(st.session_state.refined_dialogue)

        if "bleu_score" in st.session_state:
            st.subheader("📏 BLEU Score")
            st.write(f"{st.session_state.bleu_score:.4f}")

        st.download_button("📥 Download Dialogue", st.session_state.refined_dialogue, file_name="refined_dialogue.txt")
