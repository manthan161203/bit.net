import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Suppress tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Page configuration
st.set_page_config(page_title="BitNet Chat", page_icon="🤖")
st.title("🤖 BitNet Chat")

model_id = "microsoft/bitnet-b1.58-2B-4T"

# Load tokenizer and model with caching
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    return tokenizer, model

with st.spinner("Loading BitNet model..."):
    tokenizer, model = load_model()

st.sidebar.info("Running on CPU. Inference may be slow due to weight unpacking.")
st.sidebar.write(f"Model: `{model_id}`")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful AI assistant."}
    ]

# Display chat history (excluding system message for cleaner UI)
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What is on your mind?"):
    # Clear visual feedback
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Prepare input for the model
    # Note: BitNet might handle chat templates differently, using standard transformers logic
    model_prompt = tokenizer.apply_chat_template(st.session_state.messages, tokenize=False, add_generation_prompt=True)
    chat_input = tokenizer(model_prompt, return_tensors="pt").to(model.device)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            chat_outputs = model.generate(
                **chat_input, 
                max_new_tokens=2024,
                pad_token_id=tokenizer.eos_token_id
            )
            response = tokenizer.decode(chat_outputs[0][chat_input['input_ids'].shape[-1]:], skip_special_tokens=True)
            st.markdown(response)
    
    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": response})