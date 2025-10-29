import streamlit as st
import requests
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import re
import random
import time
from math import exp
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# --- 1. INITIAL SETUP & CONFIGURATION ---

# 1.1 Streamlit Page Config
st.set_page_config(
    page_title="Module 1: AI + Prompt Engineering Foundations (Cerebras)",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 1.2 Session State Initialization 
if 'progress' not in st.session_state:
    st.session_state.progress = {f'H{i}': '🔴' for i in range(1, 6)} 
    st.session_state.journal = []
    st.session_state.lab_results = {}
    st.session_state.current_tab = 'H1'
    st.session_state.guidance = "Welcome! Select a lab tab (H1-H5) to begin your module." 
    st.session_state.h1_step = 1 
    st.session_state.h4_results = pd.DataFrame() 
if 'assistant_chat_history' not in st.session_state:
    st.session_state.assistant_chat_history = []
if 'last_assistant_call' not in st.session_state:
    st.session_state.last_assistant_call = 0


# 1.3 AI Model API Configuration (Provider-Agnostic)
AI_API_URL = "https://api.cerebras.ai/v1/chat/completions"
API_KEY_NAME = "CEREBRAS_API_KEY"

# --- MODEL SELECTION (llama3.1-8b removed) ---
DEFAULT_MODEL = "qwen-3-32b" 
MODEL_OPTIONS = [
    DEFAULT_MODEL, 
    "gpt-oss-120b", 
    "llama-4-scout-17b-16e-instruct",
    "llama-3.3-70b" 
]
# -----------------------------

# --- 1.4 CUSTOM STREAMLIT STYLING (Theme: Cyber Gold) ---
STYLING = """
<style>
/* Main Background and Text */
.stApp {
    background-color: #FFFDF7; 
    color: #333; 
    font-family: Inter, sans-serif;
}

/* Sidebar and Panel Colors */
.st-emotion-cache-1c9v61q { 
    background-color: #f0f0f0;
    border-right: 2px solid #0d47a1; /* Deep Blue Accent */
}

/* Assistant Messages (For dynamic chat history) */
.assistant-message {
    background-color: #e0f7fa; /* Light cyan for assistant */
    color: #004d40;
    padding: 10px;
    border-radius: 8px;
    margin-bottom: 5px;
    border-left: 3px solid #008080;
}
.user-message {
    background-color: #fff9e0; /* Light yellow for user */
    color: #795548;
    padding: 10px;
    border-radius: 8px;
    margin-bottom: 5px;
    border-left: 3px solid #B8860B;
}

/* Highlight for the current step button */
.stButton>button[kind="primary"] {
    border: 3px solid #FF5733 !important; /* Bright orange highlight */
    animation: pulse 1.5s infinite;
}

@keyframes pulse {
    0% {
        box-shadow: 0 0 0 0 rgba(255, 87, 51, 0.7);
    }
    70% {
        box-shadow: 0 0 0 10px rgba(255, 87, 51, 0);
    }
    100% {
        box-shadow: 0 0 0 0 rgba(255, 87, 51, 0);
    }
}

/* Card/Panel Styling (Glass Effect, Shadows) */
div[data-testid*="stVerticalBlock"], div[data-testid*="stHorizontalBlock"], .stTextInput, .stTextArea, .stSelectbox {
    border-radius: 15px;
    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
    background: rgba(255, 255, 255, 0.9); /* Subtle Glass Effect */
    border: 1px solid #eee;
    padding: 10px;
}

/* Primary Buttons (Glass/Glow Effect) */
.stButton>button {
    background-color: #008080; /* Teal/Neon Accent */
    color: white !important;
    border: none;
    border-radius: 8px;
    transition: all 0.3s ease;
    font-weight: 600;
    box-shadow: 0 0 10px rgba(0, 128, 128, 0.5); /* Neon Glow */
}
.stButton>button:hover {
    background-color: #006666;
    box-shadow: 0 0 15px rgba(0, 128, 128, 0.8);
}
/* Title Styling */
.title-header {
    color: #0d47a1;
    font-weight: 800;
    font-size: 38px;
}
</style>
"""
st.markdown(STYLING, unsafe_allow_html=True)


# --- 2. UTILITY & ANALYSIS FUNCTIONS ---

def get_progress_badge(key):
    return st.session_state.progress.get(key, '🔴')

def get_progress_percent():
    """Calculates module completion percentage."""
    completed_count = sum(1 for status in st.session_state.progress.values() if status == '🟢')
    total_count = len(st.session_state.progress)
    return int((completed_count / total_count) * 100)

def update_progress(key, status):
    """Sets the completion status for a specific lab key."""
    st.session_state.progress[key] = status
    
def update_guidance(message):
    """Updates the dynamic instruction message."""
    st.session_state.guidance = message

def calculate_coherence_score(text):
    """
    Calculates a simulated Coherence Score (0-100).
    Formula: Rewards word count and word complexity (long words), penalizes extreme brevity.
    """
    word_count = len(text.split())
    long_word_count = len([w for w in text.split() if len(w) > 6])
    score = 40 + (word_count / 10) + (long_word_count * 2)
    return min(100, max(10, int(score)))

def analyze_text_metrics(text):
    """
    Calculates tokens, length, and Flesch Reading Ease (Readability Score).
    """
    tokens = text.split()
    word_count = len(tokens)
    syllable_count = sum(len(re.findall('[aeiouy]+', w.lower())) for w in tokens)
    sentence_count = len(re.split(r'[.!?]+', text))
    
    if word_count == 0 or sentence_count == 0:
        flesch_score = 100 
    else:
        # Standard Flesch Formula (simplified constant)
        flesch_score = 206.835 - 1.015 * (word_count / sentence_count) - 84.6 * (syllable_count / word_count)
    
    return {
        "tokens": len(tokens),
        "flesch_score": max(0, min(100, int(flesch_score))),
        "text_length": len(text)
    }

def save_to_journal(title, prompt, result, metrics=None):
    """Saves the lab result and reflection to the session journal."""
    st.session_state.journal.append({
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'lab': st.session_state.current_tab,
        'title': title,
        'prompt': prompt,
        'result': result,
        'metrics': metrics or {}
    })
    
def explain_llm_output(output):
    """Simulated LLM summary for the learner."""
    return "The AI Instructor has summarized the output: This response demonstrates the model's ability to generate text based on the given prompt. Analyze the metrics below to see its quality!"

def get_h2_explanation(role, tone, metrics):
    """
    Generates an AI explanation specific to the H2 Role Prompt Lab, 
    including analysis of the generated metrics.
    """
    
    flesch_score = metrics.get('flesch_score', 50)
    word_count = metrics.get('tokens', 0)
    
    base_explanation = f"The model successfully adopted the persona of a **{role}** using a **{tone}** tone. "
    
    if "Skeptical Professor" in role or "Formal" in tone:
        base_explanation += "The role constraint prioritized **detailed, complex reasoning**, which aligns with the observed "
        
    elif "Child's Book Author" in role or "Whimsical" in tone:
        base_explanation += "The model focused on **simple language and clarity** to match the persona, which should result in a higher readability score. "

    elif "Urgent" in tone:
        base_explanation += "The tone constraint enforced **short, direct sentences and focused length**, which contributes to the word count."
        
    # --- Dynamic Metric Analysis ---
    
    # 1. Readability Analysis
    if flesch_score < 30:
        base_explanation += f"Critically, the Flesch Readability Score is only **{flesch_score}/100**. This low score confirms the output uses **long sentences and complex vocabulary** (high word complexity), as is expected from a highly technical or academic persona like the **{role}**."
    elif flesch_score > 70:
        base_explanation += f"The high Flesch Readability Score of **{flesch_score}/100** indicates the language is **simple and easy to understand**, consistent with a less formal tone or audience."
    else:
        base_explanation += f"The Flesch score of **{flesch_score}/100** suggests a moderate, professional complexity, striking a balance between detail and accessibility."
        
    # 2. Length Analysis (optional)
    if word_count > 150:
        base_explanation += f" Furthermore, the **{word_count} words** used indicates a comprehensive, verbose response, which is common when the role encourages high detail (like a professor or analyst)."
    elif word_count < 50:
        base_explanation += f" The brevity of **{word_count} words** shows the model strictly adhered to the tone and max token limit, focusing only on the core answer."
    
    return base_explanation + " This lab highlights how role and style constraints fundamentally shift not just the content, but the measurable linguistic complexity of the output."


def llm_call_cerebras(messages, model=DEFAULT_MODEL, max_tokens=256, temperature=0.7):
    """Handles the secure API call to the AI Model provider with process explanation."""
    
    API_READ_TIMEOUT = 60 # Increased to 60 seconds to mitigate 'Read timed out' errors
    
    try:
        api_key = st.secrets[API_KEY_NAME] 
    except KeyError:
        if st.session_state.current_tab != 'Assistant':
             st.error(f"API Error: {API_KEY_NAME} not configured in .streamlit/secrets.toml.")
        return {"error": f"API Error: {API_KEY_NAME} not configured in .streamlit/secrets.toml."}

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}

    # START: Process Explanation (Process is described dynamically)
    if st.session_state.current_tab in [f'H{i}' for i in range(1, 6)]:
        log_container = st.container(border=True)
        log_container.subheader("Processing Output Steps")
        
        steps = [
            ("✅ Input Tokenized & Sent to AI Provider", "Your input is converted to tokens and securely transferred to the processing cluster."),
            ("💻 Model (LLM) Processing", f"The chosen LLM ({model}) is loaded and the forward pass calculation begins."),
            ("➡️ Generating Output Tokens Sequentially", "The model predicts the next token until the maximum length is reached."),
            ("✨ Final Response Compilation", "Output tokens are reassembled into human-readable text."),
        ]
        
        for i, (msg, detail) in enumerate(steps):
            log_container.markdown(f"**Step {i+1}**: {msg}")
            log_container.caption(detail)
            time.sleep(0.05) 

    # Final API Call
    start_time = time.time() 
    try:
        response = requests.post("https://api.cerebras.ai/v1/chat/completions", json=payload, headers=headers, timeout=API_READ_TIMEOUT)
        end_time = time.time() 
        
        if response.status_code != 200:
            error_detail = response.json().get("message", response.text[:100])
            if st.session_state.current_tab in [f'H{i}' for i in range(1, 6)]:
                log_container.error("🚨 API Connection Failed. Check your API Key and Model status.")
            return {"error": f"API Call Failed ({response.status_code}): {error_detail}"}

        data = response.json()
        content = data["choices"][0]["message"]["content"]
        tokens_generated = len(content.split()) 
        tokens_used = data.get("usage", {}).get("total_tokens", tokens_generated)

        if st.session_state.current_tab in [f'H{i}' for i in range(1, 6)]:
            log_container.success("✅ Response received successfully.")
        
        time_to_generate = end_time - start_time
        throughput_tps = tokens_generated / time_to_generate if time_to_generate > 0 else 0

        return {
            "content": content, 
            "model": model, 
            "tokens_used": tokens_used,
            "latency": time_to_generate,
            "throughput_tps": throughput_tps
        }

    except requests.exceptions.RequestException as e:
        if st.session_state.current_tab in [f'H{i}' for i in range(1, 6)]:
            log_container.error(f"🚨 Network Error: {e}")
        return {"error": f"API Call Failed: {e}"}


# --- 3. LAB DATA GENERATORS ---
def generate_ai_timeline_data():
    data = {'Milestone': ['Turing Test Proposed', 'Dartmouth Workshop (AI Coined)', 'Perceptron Invented', 'AI Winter Begins', 'Backpropagation Refined', 'Deep Blue Defeats Kasparov', 'ImageNet & Deep Learning Boom', 'AlphaGo Defeats Lee Sedol', 'Transformer Architecture (Attention)', 'Large Language Models (LLMs)'],
            'Year': [1950, 1956, 1957, 1974, 1986, 1997, 2012, 2016, 2017, 2023],
            'Type': ['Theory', 'Foundation', 'ML', 'Funding', 'DL', 'ML', 'DL', 'DL', 'Architecture', 'Application']}
    return pd.DataFrame(data)

def generate_kmeans_data(n_samples=300, n_clusters=4):
    X, y = make_blobs(n_samples=n_samples, centers=n_clusters, random_state=42)
    df = pd.DataFrame(X, columns=['Feature A', 'Feature B'])
    df['True Cluster'] = y
    return df


# --- 4. LAB IMPLEMENTATION FUNCTIONS (H1 - H5) ---

def render_lab1():
    st.header("H1: Exploring Your First Prompt 🚀")
    st.markdown("##### **Goal:** Learn how to write a single, effective prompt and observe the **AI Model's** response behavior.")
    
    with st.expander("📝 Instructions: Definition & Process", expanded=True):
        st.markdown("""
        **What is a Prompt?** A prompt is simply the *input* (text, command, or question) you provide to a Large Language Model (LLM) to guide its output and define its task.

        **Metrics Explained:**
        * **Latency (Time-to-Generate):** The total time (in seconds) it takes from sending the request to receiving the final byte of the response. **(Lower is better)**.
        * **Throughput (TPS):** The number of Tokens Generated Per Second (Tokens / Latency). **(Higher is better)**.

        **Action (Step 1):** Write your prompt in the text area below. Try a clear command like: **"Explain how gravity works in a single paragraph."**
        **Action (Step 2):** Click the **Run Prompt** button to execute the task on the AI Model.
        **Learning Outcome:** You will see the AI's response and understand the direct relationship between your input and the model's output.
        """)
    
    if 'h1_result' not in st.session_state:
        st.session_state.h1_result = None
    if 'h1_reflection' not in st.session_state:
        st.session_state.h1_reflection = ""
    
    # --- Step 1: Input and Parameters ---
    
    user_prompt = st.text_area(
        "Enter Your Prompt:", 
        value="Explain AI to a 5th grader.", 
        height=150, 
        key='h1_user_prompt'
    )
    
    # --- Performance Metrics DataFrame Initialization (H1 specific) ---
    if 'h1_performance_df' not in st.session_state:
        st.session_state.h1_performance_df = pd.DataFrame(columns=['Metric', 'Value'])
    
    # --- Model Selection (Use a clear model selection for the lab) ---
    with st.expander("⚙️ Adjust Model Parameters"):
        col1, col2, col3 = st.columns(3)
        with col1:
            # Use the defined MODEL_OPTIONS list
            model = st.selectbox("Model:", MODEL_OPTIONS, index=0, key='h1_model')
        with col2:
            temp = st.slider("Temperature (Creativity):", 0.0, 1.0, 0.7, 0.05, key='h1_temp')
        with col3:
            max_t = st.slider("Max Tokens:", 50, 512, 250, key='h1_max_t')
            
    st.markdown("---")
    
    # --- Step 2: Run and Execute ---
    if st.button("Run Prompt on AI Model", key='h1_run', type='primary'):
        if not user_prompt.strip():
            st.warning("Please enter a prompt before running.")
            return

        with st.spinner("Executing prompt and fetching results..."):
            messages = [{"role": "user", "content": user_prompt}]
            result = llm_call_cerebras(
                messages, 
                model=model, 
                temperature=temp, 
                max_tokens=max_t
            )
        
        st.session_state.h1_result = result
        st.session_state.h1_reflection = "" # Reset reflection
        
        if 'content' in result:
            # Calculate metrics and store in DataFrame
            metrics = analyze_text_metrics(result['content'])
            
            data = {
                'Metric': ['Model', 'Latency (s)', 'Throughput (TPS)', 'Tokens Generated', 'Flesch Readability'],
                'Value': [
                    result['model'], 
                    f"{result['latency']:.3f}", 
                    f"{result['throughput_tps']:.2f}", 
                    metrics['tokens'], 
                    metrics['flesch_score']
                ]
            }
            st.session_state.h1_performance_df = pd.DataFrame(data)

            update_guidance("✅ Step 2 Complete! Now, observe the AI's response and performance metrics.")
        
        update_progress('H1', '🟢')
        st.rerun()

    # --- Step 3: Output and Reflection ---
    if st.session_state.h1_result:
        st.subheader("AI Model Response")
        
        if 'content' in st.session_state.h1_result:
            st.code(st.session_state.h1_result['content'], language='markdown')
            
            # --- Performance Table ---
            st.subheader("Performance Metrics and Analysis")
            st.dataframe(st.session_state.h1_performance_df.set_index('Metric'), use_container_width=True)

            # AI Instructor Summary
            metrics_df = st.session_state.h1_performance_df.set_index('Metric')
            latency = metrics_df.loc['Latency (s)', 'Value']
            throughput = metrics_df.loc['Throughput (TPS)', 'Value']
            
            summary = explain_llm_output(st.session_state.h1_result['content'])
            st.subheader("🧠 AI Instructor Summary")
            st.success(f"{summary}\n\n**Performance Insight:** This model generated **{metrics_df.loc['Tokens Generated', 'Value']} tokens** in **{latency} seconds**, resulting in a strong throughput of **{throughput} TPS**.")
            
            st.markdown("---")
            st.subheader("4. Your Reflection & Insights")
            
            st.session_state.h1_reflection = st.text_area(
                "What did you notice about the response tone, detail, or structure? (e.g., 'The model used simple words because I asked it to explain to a 5th grader.')",
                value=st.session_state.h1_reflection,
                height=100,
                key='h1_reflection_input'
            )
            
            if st.button("Save Insight & Complete Lab H1", key='h1_save_complete'):
                save_to_journal("First Prompt Exploration", user_prompt, st.session_state.h1_result, {"reflection": st.session_state.h1_reflection})
                update_progress('H1', '🟢')
                update_guidance("🎉 H1 Lab Complete! Move to the **H2: Role Prompt Designer** tab to learn about persona-based prompting.")
                st.success("Insight saved! Your progress has been updated.")
                st.rerun()
                
        elif 'error' in st.session_state.h1_result:
            st.error(st.session_state.h1_result['error'])
        
def render_lab2():
    st.header("H2: Role Prompt Designer 🎭")
    st.markdown("##### **Definition:** A **Role Prompt** assigns a persona, expertise, or identity to the LLM (e.g., 'Act as a doctor').")
    st.markdown("##### **Goal:** Learn how **role**, **tone**, and **style** constraints steer the LLM's personality and output.")
    
    with st.expander("📝 Instructions: Definition & Process", expanded=False):
        st.markdown("""
        **Action:** Select a professional role and an emotional tone. Then, click **Run Role Prompt** to submit your query.
        **Output:** An LLM response filtered through the selected persona and tone.
        **Learning Outcome:** Understand that defining a **role** forces the LLM to adopt a specific knowledge profile, improving relevance and style and helping the model understand intent.
        """)

    roles = ["Security Analyst", "Child's Book Author", "Skeptical Professor", "Ancient Philosopher"]
    tones = ["Formal", "Casual", "Urgent", "Whimsical"]
    
    col1, col2 = st.columns(2)
    with col1:
        role = st.selectbox("Choose a Role:", roles, key='h2_role')
    with col2:
        tone = st.selectbox("Choose a Tone/Style Constraint:", tones, key='h2_tone')
    
    user_query = st.text_input("Your Base Query:", "Explain the importance of quantum computing.", key='h2_query')
    
    full_prompt = f"Act as a **{role}**. Respond in a **{tone}** tone. Based on these instructions, address the following query: '{user_query}'"

    st.markdown("---")
    st.subheader("Final Prompt Construction:")
    st.code(full_prompt, language='markdown')

    if st.button("Run Role Prompt (Step 1)", key='h2_run', type='primary'):
        if not user_query.strip():
            st.warning("Please enter a base query.")
            return
            
        with st.spinner("Executing role-constrained prompt..."):
            result = llm_call_cerebras([{"role": "user", "content": full_prompt}], max_tokens=250, model=DEFAULT_MODEL)
            st.session_state.h2_result = result
            
            if 'content' in result:
                st.session_state.h2_metrics = analyze_text_metrics(result['content'])
                save_to_journal(f"Role Test: {role} in {tone} tone", full_prompt, result, st.session_state.h2_metrics)
                update_progress('H2', '🟢')
                update_guidance("✅ H2 Complete! Analyze how the tone shifted in the response.")
            else:
                st.error(result['error'])

    if 'h2_result' in st.session_state and 'content' in st.session_state.h2_result:
        st.subheader("Step 2: Analysis & Visualization")
        
        col_res, col_vis = st.columns([2, 1])
        with col_res:
            st.info("LLM Response (Observe the role/tone shift):")
            st.code(st.session_state.h2_result['content'], language='markdown')
            
            # AI Explanation for H2
            st.subheader("🧠 AI Instructor Explanation")
            # PASS THE METRICS to the explanation function
            ai_explanation = get_h2_explanation(role, tone, st.session_state.h2_metrics)
            st.success(ai_explanation)
            
        
        with col_vis:
            metrics = st.session_state.h2_metrics
            st.metric("Words Used", metrics['tokens'])
            st.metric("Flesch Readability", f"{metrics['flesch_score']}/100")
            
            st.markdown("##### Tone Histogram (Simulated)")
            tone_score = {"Formal": 90, "Casual": 30, "Urgent": 70, "Whimsical": 50}[tone]
            st.bar_chart({"Score": [tone_score]}, use_container_width=True)
            
    elif 'h2_result' in st.session_state and 'error' in st.session_state.h2_result:
        st.error(st.session_state.h2_result['error'])


def render_lab3():
    st.header("H3: Temperature & Context Lab 🌡️")
    st.markdown("##### **Definition:** **Temperature** controls the randomness (creativity) of the LLM's output.")
    st.markdown("##### **Goal:** Explore stochasticity (creativity) and context sensitivity.")

    with st.expander("📝 Instructions: Action, Output, & Learning", expanded=False):
        st.markdown("""**Action:** Run the same prompt multiple times while changing the **Temperature** (randomness) and **Context Length** (max output size). **Output:** A history table and a scatter plot visualizing the results of each experiment. **Learning Outcome:** Develop intuition for the **creativity vs. coherence** trade-off controlled by the Temperature parameter.""")

    st.subheader("1. Parameters and Prompt")
    col1, col2 = st.columns(2)
    with col1:
        temp = st.slider("Temperature (0.0 - 1.5):", 0.0, 1.5, 0.7, 0.1, key='h3_temp')
        st.caption("0.0 = Predictable/Coherent; 1.5 = Highly Creative/Random.")
    with col2:
        context_options = ['Short (50 tokens)', 'Medium (200 tokens)', 'Long (500 tokens)']
        context_len = st.radio("Context Length (Simulated):", context_options, key='h3_context')
        
        match = re.search(r'\((\d+)', context_len)
        max_t = int(match.group(1)) if match else 200

    
    prompt = st.text_area("Base Prompt:", "Write a short poem about a self-driving car encountering rain for the first time.", key='h3_prompt')
    
    if st.button("Run Experiment (Step 1)", key='h3_run', type='primary'):
        if 'h3_history' not in st.session_state:
            st.session_state.h3_history = []
        
        with st.spinner(f"Running prompt at Temp={temp}, Max Tokens={max_t}..."):
            messages = [{"role": "user", "content": prompt}]
            result = llm_call_cerebras(messages, temperature=temp, max_tokens=max_t, model=DEFAULT_MODEL) 
            
            if 'content' in result:
                metrics = analyze_text_metrics(result['content'])
                st.session_state.h3_history.append({
                    'id': len(st.session_state.h3_history) + 1,
                    'temp': temp,
                    'context': context_len.split('(')[0].strip(),
                    'tokens': metrics['tokens'],
                    'flesch': metrics['flesch_score'],
                    'result': result['content']
                })
                save_to_journal(f"Temp/Context Exp: T={temp}, C={context_len}", prompt, result)
                update_progress('H3', '🟢') 
                update_guidance("✅ H3 Complete! Analyze the plot to see the creativity trade-off.")
            else:
                 st.error(result['error'])


    if 'h3_history' in st.session_state and st.session_state.h3_history:
        st.subheader("Step 2: Experiment History")
        
        history_df = pd.DataFrame(st.session_state.h3_history)
        st.dataframe(history_df[['id', 'temp', 'context', 'tokens', 'flesch']], hide_index=True, use_container_width=True)

        st.subheader("Step 3: Visualization - Coherence vs Diversity")
        fig = px.scatter(history_df, x='temp', y='tokens', size='flesch', color='context', 
                             title="Token Count vs. Temperature (Bubble size = Readability)",
                             labels={'temp': 'Temperature (Diversity)', 'tokens': 'Token Count'},
                             color_discrete_sequence=['#B8860B', '#0d47a1', '#008080'])
        st.plotly_chart(fig, use_container_width=True)
        
        # --- AI Instructor Graph Summary ---
        st.subheader("🧠 AI Instructor Graph Summary")
        if not history_df.empty:
            max_temp = history_df['temp'].max()
            
            summary_message = f"""
            The **Temperature vs. Token Count** chart visualizes the effects of stochasticity (randomness). 
            * **X-axis (Temperature):** Shows how much randomness was introduced. As this value increases, the model's choices become more diverse, leading to **greater output variation**.
            * **Y-axis (Token Count):** Shows the length of the response.
            * **Bubble Size (Readability/Flesch Score):** The size of the bubble indicates how easy the text is to read.
            
            **Key Insight:** Generally, experiments run with **low temperatures** (closer to 0.0) cluster lower on the diversity scale but often have **higher readability** (larger bubbles), while experiments run with **high temperatures** (closer to {max_temp}) show more unique results but sometimes lower coherence. This highlights the fundamental trade-off in generative AI.
            """
            st.success(summary_message)


def render_lab4():
    st.header("H4: Multi-Prompt Comparison Studio ⭐")
    st.markdown("##### **Definition:** **Comparison Studio** uses parallel execution to score different prompts against the same criteria.")
    st.markdown("##### **Goal:** Compare and objectively score the output quality of up to three different prompts.")

    with st.expander("📝 Instructions: Action, Output, & Learning", expanded=False):
        st.markdown("""**Action:** Enter 2 or 3 distinct prompts. Click **Run All Prompts in Parallel**. **Output:** A comparative table with auto-calculated metrics (Tokens, Coherence Score) and a space for your manual rating. **Learning Outcome:** Evaluate which prompt design is objectively superior based on metrics and your own manual scoring.""")

    if 'h4_prompts' not in st.session_state:
        st.session_state.h4_prompts = ["", "", ""]
    if 'h4_results' not in st.session_state or st.session_state.h4_results is None:
        st.session_state.h4_results = pd.DataFrame() 

    st.subheader("1. Enter Prompts (Max 3)")
    cols = st.columns(3)
    for i in range(3):
        st.session_state.h4_prompts[i] = cols[i].text_area(f"Prompt {i+1}", st.session_state.h4_prompts[i], height=150, key=f'h4_p{i}')

    col_run, col_param = st.columns([1, 2])
    with col_param:
        temp = st.slider("Temperature:", 0.0, 1.0, 0.5, 0.1, key='h4_temp')
        max_t = st.slider("Max Tokens:", 100, 400, 250, key='h4_max_t')

    if col_run.button("Run All Prompts in Parallel (Step 2)", key='h4_run', type='primary'):
        valid_prompts = [p for p in st.session_state.h4_prompts if p.strip()]
        if not valid_prompts:
            st.warning("Please enter at least one prompt.")
            return

        results = []
        for i, prompt in enumerate(valid_prompts):
            with st.spinner(f"Running Prompt {i+1}...") as s:
                messages = [{"role": "user", "content": prompt}]
                result = llm_call_cerebras(messages, temperature=temp, max_tokens=max_t, model=DEFAULT_MODEL)
                
                if 'content' in result:
                    metrics = analyze_text_metrics(result['content'])
                    coherence = calculate_coherence_score(result['content'])
                    results.append({
                        'Prompt ID': i + 1,
                        'Prompt Text': prompt[:40] + "...",
                        'Response': result['content'],
                        'Tokens Used': result['tokens_used'],
                        'Length (Words)': metrics['tokens'],
                        'Coherence Score': coherence,
                        'Manual Rating': 3 
                    })
                else:
                     st.error(f"Prompt {i+1} failed: {result['error']}")
            
            # --- RATE LIMIT MITIGATION ---
            if i < len(valid_prompts) - 1: # Don't sleep after the last prompt
                st.info(f"Adding 3-second delay to avoid rate-limiting (429 errors). Resuming in 3 seconds...")
                time.sleep(3)
            # ---------------------------------
        
        if results:
            st.session_state.h4_results = pd.DataFrame(results) 
            update_progress('H4', '🟡') 
            update_guidance("➡️ H4: Review the table below and manually rate each response (Step 3).")

    if not st.session_state.h4_results.empty:
        st.subheader("3. Comparative Results")
        
        results_df = st.session_state.h4_results.copy() 
        
        rating_inputs = []
        for i, row in results_df.iterrows():
            rating = st.slider(f"Prompt {row['Prompt ID']} Rating (1-5):", 1, 5, int(row.get('Manual Rating', 3)), key=f'h4_rate_{row["Prompt ID"]}')
            rating_inputs.append(rating)
        results_df['Manual Rating'] = rating_inputs

        st.dataframe(results_df[['Prompt Text', 'Tokens Used', 'Length (Words)', 'Coherence Score', 'Manual Rating']], use_container_width=True)

        # --- Post-Comparison Summary ---
        if results_df['Manual Rating'].sum() > 0:
            best_prompt_id = results_df.loc[results_df['Manual Rating'].idxmax()]['Prompt ID']
            best_coherence = results_df['Coherence Score'].max()
            
            st.markdown("---")
            st.subheader("Summary: Response Quality Insight")
            st.info(f"""
            Based on your ratings and the calculated metrics:
            * **Best Performer:** Prompt **#{best_prompt_id}** was manually rated the highest.
            * **Key Metric:** The highest calculated Coherence Score was **{best_coherence}**.
            * **Conclusion:** The prompt that yields the highest scores is typically the most specific and clearly structured. Use the best-performing prompt's structure as a template for future labs.
            """)
        # --- End Post-Comparison Summary ---

        if st.button("Save & Complete Lab H4", key='h4_complete'):
            for _, row in results_df.iterrows():
                save_to_journal(f"H4 Comparison Prompt {row['Prompt ID']}", row['Prompt Text'], row['Response'], 
                                 {'Coherence Score': row['Coherence Score'], 'Manual Rating': row['Manual Rating']})
            update_progress('H4', '🟢')
            st.success("H4 Complete! Results saved to Learning Journal.")
            update_guidance("✅ H4 Complete! Move to H5: Prompt Optimization Challenge.")
            st.experimental_rerun()


def render_lab5():
    st.header("H5: Prompt Optimization Challenge 🎯")
    st.markdown("##### **Definition:** **Prompt Optimization** is the process of iteratively refining a prompt until it meets predefined, measurable quality standards.")
    st.markdown("##### **Goal:** Iteratively refine a prompt to meet specific, measurable performance thresholds.")

    with st.expander("📝 Instructions: Action, Output, & Learning", expanded=False):
        st.markdown("""
        **Action:** Write your prompt to meet the two targets (Readability and Tokens). Click **Run Attempt** to test.
        **Output:** Live feedback on whether your prompt passed the metrics.
        **Learning Outcome:** Practice systematic prompt refinement to achieve objective, metric-based goals.
        """)

    target_flesch = st.slider("Target Readability (Flesch Score):", 60, 100, 80, key='h5_target_flesch')
    target_tokens = st.slider("Max Token Limit:", 20, 100, 50, key='h5_target_tokens')

    st.subheader("1. The Challenge")
    st.info(f"Challenge: Write a prompt that results in a definition of AI with a Readability Score **above {target_flesch}** and a Token Count **below {target_tokens}**.")
    
    prompt = st.text_area("Your Optimization Prompt:", "Define Artificial Intelligence.", height=100, key='h5_prompt')
    
    if 'h5_attempts' not in st.session_state:
        st.session_state.h5_attempts = []
    
    if st.button("Run Attempt (Step 2)", key='h5_run', type='primary'):
        if not prompt.strip():
            st.warning("Please enter a prompt.")
            return

        with st.spinner("Running optimization attempt..."):
            messages = [{"role": "user", "content": prompt}]
            result = llm_call_cerebras(messages, max_tokens=target_tokens, model=DEFAULT_MODEL) 
            
            if 'content' in result:
                metrics = analyze_text_metrics(result['content'])
                
                # Check metrics against targets
                passed = metrics['flesch_score'] >= target_flesch and metrics['tokens'] <= target_tokens
                
                st.session_state.h5_attempts.append({
                    'id': len(st.session_state.h5_attempts) + 1,
                    'prompt': prompt,
                    'flesch_score': metrics['flesch_score'],
                    'tokens': metrics['tokens'],
                    'passed': passed,
                    'response': result['content']
                })
                save_to_journal(f"Optimization Attempt {len(st.session_state.h5_attempts)}", prompt, result, metrics)
                
                if passed:
                    update_progress('H5', '🟢')
                    update_guidance(f"🥳 Success! H5 complete on attempt {len(st.session_state.h5_attempts)}.")
                else:
                    update_progress('H5', '🟡')
                    update_guidance("🟡 H5: Attempt failed. Analyze the metrics below and adjust your prompt (Step 1).")
            else:
                 st.error(result['error'])


    if st.session_state.h5_attempts:
        st.subheader("3. Optimization Trajectory")
        last_attempt = st.session_state.h5_attempts[-1]
        
        if last_attempt['passed']:
            st.balloons()
            st.success(f"🥳 CHALLENGE PASSED on Attempt {last_attempt['id']}! Metrics met the target.")
        else:
            st.error(f"❌ Attempt {last_attempt['id']} Failed. Adjust your prompt and try again.")
        
        col_f, col_t = st.columns(2)
        col_f.metric(f"Current Readability (Target > {target_flesch})", last_attempt['flesch_score'], 
                     delta_color='normal' if last_attempt['flesch_score'] >= target_flesch else 'inverse')
        col_t.metric(f"Current Tokens (Target < {target_tokens})", last_attempt['tokens'],
                     delta_color='normal' if last_attempt['tokens'] <= target_tokens else 'inverse')
        
        st.markdown("---")
        st.code(last_attempt['response'], language='markdown')
        
        df_attempts = pd.DataFrame(st.session_state.h5_attempts)
        fig = px.line(df_attempts, x='id', y=['flesch_score', 'tokens'], 
                      title="Performance Metrics Over Attempts",
                      labels={'id': 'Attempt Number', 'value': 'Score/Tokens'},
                      color_discrete_map={'flesch_score': '#0d47a1', 'tokens': '#B8860B'})
        st.plotly_chart(fig, use_container_width=True)


# --- 5. AI ASSISTANT FUNCTION (LLM INTEGRATED) ---

def render_ai_assistant_sidebar():
    """Renders the persistent AI Assistant and Learning Journal in the sidebar, now powered by LLM."""
    
    st.sidebar.markdown('<div class="sidebar-assistant">', unsafe_allow_html=True)
    
    # 1. Guidance Message Display
    guidance_message = st.session_state.get('guidance', "Welcome! Select a lab tab (H1-H5) to begin your module.")
    st.sidebar.markdown('**Current Task:**')
    st.sidebar.info(guidance_message)
    st.sidebar.markdown('</div>', unsafe_allow_html=True)

    # 2. AI Assistant Chat Interface
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### Ask the AI Assistant")
    
    # Predefined System Context for the AI Assistant
    SYSTEM_PROMPT = """
    You are the **AI Instructor Assistant** for the 'AI + Prompt Engineering Foundations' lab module. 
    Your role is to guide the user through the lab activities, define concepts, and provide troubleshooting tips related to the dashboard.
    
    The user is currently working on a Streamlit application with the following lab tabs:
    - H1: Exploring Your First Prompt (Goal: Observe basic prompt output, latency, and throughput).
    - H2: Role Prompt Designer (Goal: Understand how 'Role' and 'Tone' constraints steer the LLM's personality).
    - H3: Temperature & Context Lab (Goal: Explore the creativity/coherence trade-off using the 'Temperature' parameter).
    - H4: Multi-Prompt Comparison Studio (Goal: Compare multiple prompts using metrics like Coherence Score and Manual Rating).
    - H5: Prompt Optimization Challenge (Goal: Iteratively refine a prompt to meet specific metric targets, like Readability and Token count).
    
    **Instructions for your response:**
    1. Be concise, encouraging, and highly relevant to the lab context.
    2. If asked 'What should I do next?' or a similar question, guide them to the next logical step based on the lab flow (H1 -> H2, H2 -> H3, etc.).
    3. If asked for a definition (e.g., 'what is prompt engineering'), provide a clear, one or two-sentence summary.
    4. Do NOT execute any external code or API calls yourself. The user will do this in the main app.
    5. Always maintain a helpful, encouraging tone.
    """
    
    user_query = st.sidebar.text_input("Ask about the module, steps, or concepts:", key="assistant_query")
    
    if st.sidebar.button("Ask Assistant", key="run_assistant"):
        
        # --- NEW ASSISTANT COOLDOWN CHECK ---
        if time.time() - st.session_state.last_assistant_call < 5:
            st.sidebar.error("Please wait 5 seconds before asking the Assistant another question to avoid rate-limiting (429 errors).")
            return
        # ------------------------------------

        if user_query:
            # Add user message to history
            st.session_state.assistant_chat_history.append({"role": "user", "content": user_query})
            
            # 1. Construct the message list for the LLM
            messages = [{"role": "system", "content": SYSTEM_PROMPT}] 
            messages.extend(st.session_state.assistant_chat_history[-4:]) # Last 4 messages for context

            # Temporarily set current_tab to a non-lab value to suppress log messages in main panel
            temp_current_tab = st.session_state.current_tab
            st.session_state.current_tab = 'Assistant'
            
            with st.spinner("Assistant is thinking..."):
                # 2. Call the LLM 
                assistant_result = llm_call_cerebras(
                    messages=messages, 
                    model=DEFAULT_MODEL, # Use the new default model
                    max_tokens=256, 
                    temperature=0.2 
                )
            
            st.session_state.current_tab = temp_current_tab # Restore current tab
            
            # --- UPDATE COOLDOWN TIMER ---
            st.session_state.last_assistant_call = time.time()
            # -----------------------------
            
            if 'content' in assistant_result:
                response = assistant_result['content']
            elif 'error' in assistant_result:
                response = f"**Assistant Error:** Sorry, I encountered an issue: {assistant_result['error']}. This may indicate an issue with your API key or the model status. The code has been updated with a longer timeout and cooldowns to mitigate rate limits. Please try again soon."
            else:
                response = "I'm experiencing a service interruption. Please try again in a moment."

            # Add assistant message to history and display
            st.session_state.assistant_chat_history.append({"role": "assistant", "content": response})
            st.rerun() 

    # 3. Display Chat History
    for message in st.session_state.assistant_chat_history:
        if message['role'] == 'user':
            st.sidebar.markdown(f'<div class="user-message">**You:** {message["content"]}</div>', unsafe_allow_html=True)
        elif message['role'] == 'assistant':
            st.sidebar.markdown(f'<div class="assistant-message">**Assistant:** {message["content"]}</div>', unsafe_allow_html=True)

    
    # 4. Reset Button
    st.sidebar.markdown("---")
    if st.sidebar.button("Reset All Lab Progress (Clear Session) ⚠️", type='secondary'):
        # Clear all session state keys
        for key in list(st.session_state.keys()):
             del st.session_state[key]

        # Initialize base state again
        st.session_state.progress = {f'H{i}': '🔴' for i in range(1, 6)} 
        st.session_state.journal = []
        st.session_state.lab_results = {}
        st.session_state.current_tab = 'H1'
        st.session_state.guidance = "Welcome! Select a lab tab (H1-H5) to begin your module." 
        st.session_state.h1_step = 1 
        st.session_state.h4_results = pd.DataFrame() 
        st.session_state.assistant_chat_history = []
        st.session_state.last_assistant_call = 0 # Reset cooldown timer
        
        st.success("Session cleared. Please refresh the browser.")
        st.rerun()


# --- 6. MAIN APPLICATION ENTRY POINT ---

def render_main_page():
    # Final App Title (Enhanced)
    st.markdown('<div class="title-header">Module 1: AI + Prompt Engineering Foundations</div>', unsafe_allow_html=True)
    st.markdown("---")

    # Tab Titles
    tab_titles = [
        "H1: Exploring Your First Prompt 🚀", "H2: Role Prompt Designer 🎭", 
        "H3: Temperature & Context Lab 🌡️", "H4: Multi-Prompt Comparison Studio ⭐", 
        "H5: Prompt Optimization Challenge 🎯"
    ]
    tabs = st.tabs(tab_titles)
    
    # Content Rendering
    with tabs[0]:
        st.session_state.current_tab = 'H1'
        render_lab1()
    with tabs[1]:
        st.session_state.current_tab = 'H2'
        render_lab2()
    with tabs[2]:
        st.session_state.current_tab = 'H3'
        render_lab3()
    with tabs[3]:
        st.session_state.current_tab = 'H4'
        render_lab4()
    with tabs[4]:
        st.session_state.current_tab = 'H5'
        render_lab5()

if __name__ == '__main__':
    render_ai_assistant_sidebar()
    render_main_page()
