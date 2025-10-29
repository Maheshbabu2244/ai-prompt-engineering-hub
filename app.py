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
    st.session_state.current_tab = 'Intro' # Changed initial tab to Intro
    st.session_state.guidance = "Welcome! Click the **🧭 Getting Started** tab to begin your module." 
    st.session_state.h1_step = 1 
    st.session_state.h4_results = pd.DataFrame() 
    st.session_state.onboarding_done = False # New state for modal
if 'assistant_chat_history' not in st.session_state:
    st.session_state.assistant_chat_history = [{"role": "assistant", "content": "👋 Hi there! I'm your AI Instructor. Ask me anything in simple terms about prompts, AI settings (like 'Temperature'), or what to do next!"}]
if 'last_assistant_call' not in st.session_state:
    st.session_state.last_assistant_call = 0


# 1.3 AI Model API Configuration (Provider-Agnostic)
AI_API_URL = "https://api.cerebras.ai/v1/chat/completions"
API_KEY_NAME = "CEREBRAS_API_KEY"

# --- MODEL SELECTION ---
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

/* Progress Tracker Styling for Sidebar */
.progress-tracker {
    padding: 10px;
    border-radius: 10px;
    background-color: #ffffff;
    border: 1px solid #ddd;
    margin-bottom: 10px;
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

def glossary_tooltip(term: str, definition: str):
    """Helper for creating clickable tooltip-like text."""
    # Added this missing function
    return f'<span title="{definition}" style="cursor: pointer; border-bottom: 1px dotted #0d47a1;">{term} ℹ️</span>'

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
    """Simulated LLM summary for the learner (Simplified)."""
    return "The AI Instructor has summarized the output: This response demonstrates the model's ability to generate text based on the given prompt. Analyze the metrics below to see its quality!"

def get_h2_explanation(role, tone, metrics):
    """
    Generates an AI explanation specific to the H2 Role Prompt Lab.
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
    
    API_READ_TIMEOUT = 60
    
    # Check for API Key (simplified for non-technical users)
    try:
        api_key = st.secrets[API_KEY_NAME] 
    except KeyError:
        if st.session_state.current_tab != 'Assistant':
             st.error(f"⚠️ **API Key Missing!** Please configure the **{API_KEY_NAME}** in your `secrets.toml` file to run the labs.")
        return {"error": f"API Error: {API_KEY_NAME} not configured."}

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}

    # START: Process Explanation (Simplified for non-technical users)
    if st.session_state.current_tab in [f'H{i}' for i in range(1, 6)]:
        log_container = st.container(border=True)
        log_container.subheader("💻 AI Model Processing Steps")
        
        steps = [
            ("✅ Input Sent", "Your prompt is securely sent."),
            ("💻 AI Model Working", f"The chosen model ({model}) is calculating the best response."),
            ("➡️ Generating Words", "The model predicts the output word by word."),
            ("✨ Response Ready", "The final text is compiled and returned to the dashboard."),
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
                 log_container.error("🚨 **Error:** The AI connection failed. Please check your API Key or try a different model.")
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
             log_container.error(f"🚨 **Network/Timeout Error:** {e}. This usually means the request took too long.")
        return {"error": f"API Call Failed: {e}"}


# --- 3. ONBOARDING & GETTING STARTED DASHBOARD ---

def show_onboarding_modal():
    """1. Onboarding Modal (First Login) - Popup version."""
    if not st.session_state.get("onboarding_done", False):
        st.session_state["onboarding_done"] = True
        st.toast("👋 Welcome to AI Prompt Engineering Explorer! Let's get started!", icon="🚀")

        with st.popover("✨ **Welcome to the Explorer Lab! Click Here to Start!** ✨", use_container_width=True):
            st.markdown("""
                ### Here's Your Guided Flow:
                1.  **Prompts (H1):** Learn what a prompt is and how the AI reacts. 💬
                2.  **Refinement (H2):** Learn to give the AI a role and tone for better results. 🎭
                3.  **Analysis (H3+):** Learn to measure and improve the quality of the AI's response. 📏
                
                Click the tabs (H1, H2, etc.) above to begin your hands-on training!
            """)
            st.progress(0.1, text="Loading Core Concepts...")


def render_getting_started():
    """
    Creates the main dashboard view detailing H1 through H5 labs with
    definitions, goals, and interactive elements.
    """
    st.markdown('<div class="title-header">🧭 Your Prompt Engineering Journey</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    st.header("💡 Module Overview: From Idea to Instruction")
    st.info("""
        This module teaches you **Prompt Engineering**—the art of talking to AI effectively. 
        We move from basic instructions (H1) and adding personality (H2) to advanced techniques like controlling creativity (**Temperature**, H3), comparing prompts (H4), and hitting objective metrics (H5).
    """)

    # --- H1: EXPLORING YOUR FIRST PROMPT ---
    st.markdown("---")
    st.subheader("⚙️ H1: Exploring Your First Prompt 🚀 (The Basics)")
    
    col_def, col_goal = st.columns([1, 1])

    with col_def:
        st.markdown("#### What You'll Explore:")
        st.markdown(f"""
            - **Definition:** A **Prompt** is the direct instruction you give the AI. 
            - **Key Concept:** You'll learn that even small changes to a prompt lead to big changes in the **Output** (the AI's response).
            - **New Terms:** Learn about {glossary_tooltip('Temperature', 'The creativity dial. Low value = predictable. High value = random/creative.')} and **Latency** (the AI's thinking time).
        """)

    with col_goal:
        st.markdown("#### The Goal:")
        st.success("""
            **Goal:** Successfully run your first prompt and understand the **direct cause-and-effect** between your words and the AI's output, including its speed.
        """)
        st.markdown("---")
        st.markdown("#### Step-by-Step Guide Preview:")
        st.progress(0.0, text="Step 1/3: Enter Prompt → Step 2/3: Run → Step 3/3: Analyze")
            
    # --- H2: ROLE PROMPT DESIGNER ---
    st.markdown("---")
    st.subheader("🎭 H2: Role Prompt Designer (Adding Personality)")
    
    col_def_2, col_goal_2 = st.columns([1, 1])

    with col_def_2:
        st.markdown("#### What You'll Explore:")
        st.markdown(f"""
            - **Definition:** A **Role Prompt** gives the AI a persona, like "Act as a pirate" or "You are a CEO."
            - **Key Concept:** By adding **Role** and **Tone** (e.g., Formal, Humorous), you control the AI's personality, leading to better, more relevant outputs.
            - **New Terms:** Learn how {glossary_tooltip('Readability Score', 'A metric (like Flesch) that measures how easy the text is for a general audience to read.')} changes based on the persona.
        """)

    with col_goal_2:
        st.markdown("#### The Goal:")
        st.success("""
            **Goal:** Master prompt structure by forcing the AI to adopt a **specific role and tone**, demonstrating that the AI's style is fully controllable and measurable.
        """)
        st.markdown("---")
        st.markdown("#### Step-by-Step Guide Preview:")
        st.progress(0.0, text="Step 1/2: Select Role/Tone → Step 2/2: Run & Analyze Persona Shift")
    
    # --- H3: TEMPERATURE & CONTEXT LAB ---
    st.markdown("---")
    st.subheader("🌡️ H3: Temperature & Context Lab (Controlling Creativity)")
    
    col_def_3, col_goal_3 = st.columns([1, 1])

    with col_def_3:
        st.markdown("#### Core Definition:")
        st.markdown(f"""
            - **Definition:** {glossary_tooltip('Temperature', 'The creativity dial. Low value = predictable. High value = random/creative.')} dictates the randomness and creativity of the AI's output (its level of "wildness").
            - **Key Concept:** You'll explore **Stochasticity** (the randomness in AI choices) by running the same prompt multiple times while changing the Temperature dial.
            - **Hands-on Action:** You'll run an experiment (like writing a poem) at various Temperature settings (e.g., 0.2, 0.7, 1.5) and visualize the results.
        """)

    with col_goal_3:
        st.markdown("#### The Goal:")
        st.success("""
            **Goal:** Develop intuition for the fundamental trade-off between **Creativity and Coherence** (does it make sense?), showing how randomness affects output quality. You'll know when to use low Temperature (for facts/code) and high Temperature (for brainstorming/poetry).
        """)
        st.markdown("---")
        st.markdown("#### Step-by-Step Guide Preview:")
        st.progress(0.0, text="Step 1/3: Set Temperature & Prompt → Step 2/3: Run Experiments → Step 3/3: Compare Outputs")
        
    # --- H4: MULTI-PROMPT COMPARISON STUDIO ---
    st.markdown("---")
    st.subheader("⭐ H4: Multi-Prompt Comparison Studio (Finding the Best Prompt)")
    
    col_def_4, col_goal_4 = st.columns([1, 1])

    with col_def_4:
        st.markdown("#### Core Definition:")
        st.markdown(f"""
            - **Definition:** **Prompt Comparison** is the simultaneous execution and objective scoring of several different prompts designed for the same objective.
            - **Key Concept:** You'll use the **Coherence Score** (how logically consistent the text is) and your own **Manual Rating** (1-5 stars) to find the best prompt.
            - **Hands-on Action:** Input 2–3 different ways to ask for the same thing (e.g., three prompts for explaining blockchain), run them in parallel, and rate the resulting outputs in a comparative table.
        """)

    with col_goal_4:
        st.markdown("#### The Goal:")
        st.success("""
            **Goal:** Determine which prompt structure is **empirically superior** by scoring outputs based on both automatic metrics and your manual judgment. You'll learn to think like a reviewer, identifying the structural elements (specificity, clarity) that consistently produce high-quality AI results.
        """)
        st.markdown("---")
        st.markdown("#### Step-by-Step Guide Preview:")
        st.progress(0.0, text="Step 1/3: Write 3 Prompts → Step 2/3: Run & Score → Step 3/3: Analyze Results Table")

    # --- H5: PROMPT OPTIMIZATION CHALLENGE ---
    st.markdown("---")
    st.subheader("🎯 H5: Prompt Optimization Challenge (Achieving Objective Control)")
    
    col_def_5, col_goal_5 = st.columns([1, 1])

    with col_def_5:
        st.markdown("#### Core Definition:")
        st.markdown(f"""
            - **Definition:** **Prompt Optimization** is the iterative process of refining a prompt until its output meets specific, measurable quality standards or **Target Thresholds**.
            - **Key Concept:** This is a test of your ability to control complexity ({glossary_tooltip('Readability', 'A metric (like Flesch) that measures how easy the text is for a general audience to read.')}) and length (**Tokens**).
            - **Hands-on Action:** Set targets (e.g., Readability > 80, Tokens < 50), write an initial prompt, run it, analyze the metrics, and refine your prompt until you pass both targets.
        """)

    with col_goal_5:
        st.markdown("#### The Goal:")
        st.success("""
            **Goal:** Iteratively refine your prompt to hit **two objective targets simultaneously**. You'll gain confidence in designing, testing, and controlling AI output independently, culminating in the completion of your Prompt Engineering Foundations Module.
        """)
        st.markdown("---")
        st.markdown("#### Step-by-Step Guide Preview:")
        st.progress(0.0, text="Step 1/4: Set Targets → Step 2/4: Write Prompt → Step 3/4: Run & Check Metrics → Step 4/4: Refine & Repeat")

    st.markdown("---")
    st.markdown("### Ready to start? Click on the **H1: First Prompt 🚀** tab above!")


# --- 4. LAB IMPLEMENTATION FUNCTIONS (H1 - H5) ---

# The bodies of the render_labX functions are kept as provided in the prompt.
def render_lab1():
    st.header("H1: Exploring Your First Prompt 🚀")
    st.markdown("##### **Goal:** Learn how to write a single, effective prompt and observe the **AI Model's** response behavior.")
    
    with st.expander("📝 Instructions: Definition & Process", expanded=True):
        st.markdown("""
        **What is a Prompt?** A **prompt** is simply the *message* or *instruction* you give to the AI to tell it what to do. Think of it as sending a text message to a highly intelligent assistant!

        **Metrics Explained (The AI's Speed):**
        * **Latency (Time-to-Generate):** How many seconds the AI takes to think and write the full answer. **(Lower is better)**.
        * **Throughput (TPS):** The speed at which the AI writes, measured in Words (or Tokens) Per Second. **(Higher is better)**.

        **Action (Step 1):** Write your prompt. Try a simple command like: **"Explain how gravity works in a single paragraph."**
        **Action (Step 2):** Click **Run Prompt** to see the magic happen!
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
    with st.expander("⚙️ Adjust AI Dials (Advanced Settings)"):
        col1, col2, col3 = st.columns(3)
        with col1:
            # Use the defined MODEL_OPTIONS list
            model = st.selectbox("Model:", MODEL_OPTIONS, index=0, key='h1_model', help="The specific AI engine you are using.")
        with col2:
            temp = st.slider("Temperature (Creativity):", 0.0, 1.0, 0.7, 0.05, key='h1_temp', help="Controls randomness. 0.0 is very predictable. 1.0 is very creative/random.")
        with col3:
            max_t = st.slider("Max Tokens (Max Length):", 50, 512, 250, key='h1_max_t', help="Sets the maximum number of words the AI can write in its response.")
            
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

                update_guidance("✅ H1 Step 2 Complete! Now, observe the AI's response and performance metrics.")
            
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
            st.success(f"{summary}\n\n**Performance Insight:** This model generated **{metrics_df.loc['Tokens Generated', 'Value']} tokens** in **{latency} seconds**, resulting in a strong speed (throughput) of **{throughput} TPS**.")
            
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
        temp = st.slider("Temperature (0.0 - 1.5):", 0.0, 1.5, 0.7, 0.1, key='h3_temp', help="Low value = predictable. High value = random/creative.")
        st.caption("0.0 = Predictable/Coherent; 1.5 = Highly Creative/Random.")
    with col2:
        context_options = ['Short (50 tokens)', 'Medium (200 tokens)', 'Long (500 tokens)']
        context_len = st.radio("Context Length (Max Words):", context_options, key='h3_context', help="This simulates setting the maximum word count for the AI's response.")
        
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
            The **Temperature vs. Token Count** chart shows how the AI's settings change the output. 
            * **X-axis (Temperature):** This shows how **creative** (random) the AI was.
            * **Y-axis (Token Count):** This shows how **long** the response was.
            * **Bubble Size (Readability):** Bigger bubbles mean the text is **easier to read**.
            
            **Key Insight:** When you set a **low temperature** (closer to 0.0), the AI is more predictable and the results are often more similar. When you use a **high temperature** (closer to {max_temp}), the outputs are very different and more creative, but sometimes less focused or "coherent."
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
        temp = st.slider("Temperature:", 0.0, 1.0, 0.5, 0.1, key='h4_temp', help="A neutral temperature for fair comparison.")
        max_t = st.slider("Max Tokens:", 100, 400, 250, key='h4_max_t', help="A consistent length ensures fair comparison.")

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
                st.info(f"Adding 3-second delay to avoid service interruption. Resuming in 3 seconds...")
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
            # Use the Manual Rating column from the DataFrame, or default to 3
            initial_rating = int(row.get('Manual Rating', 3)) 
            rating = st.slider(f"Prompt {row['Prompt ID']} Rating (1=Poor, 5=Excellent):", 1, 5, initial_rating, key=f'h4_rate_{row["Prompt ID"]}')
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
            * **Key Metric:** The highest calculated **Coherence Score** (how much it makes sense) was **{best_coherence}**.
            * **Conclusion:** The prompt that won is likely the one that was most **specific** and **clearly structured**. Use that winning structure as a template!
            """)
        # --- End Post-Comparison Summary ---

        if st.button("Save & Complete Lab H4", key='h4_complete'):
            for _, row in results_df.iterrows():
                # Save each prompt comparison result to the journal
                save_to_journal(f"H4 Comparison Prompt {row['Prompt ID']}", st.session_state.h4_prompts[row['Prompt ID']-1], {'content': row['Response']}, 
                                 {'Coherence Score': row['Coherence Score'], 'Manual Rating': row['Manual Rating']})
            update_progress('H4', '🟢')
            st.success("H4 Complete! Results saved to Learning Journal.")
            update_guidance("✅ H4 Complete! Move to H5: Prompt Optimization Challenge.")
            st.rerun()


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

    target_flesch = st.slider("Target Readability (Easy-to-Read Score):", 60, 100, 80, key='h5_target_flesch', help="A higher Flesch Score means the text is easier for a general audience to read.")
    target_tokens = st.slider("Max Token Limit (Max Words):", 20, 100, 50, key='h5_target_tokens', help="Your response must be shorter than this number of words.")

    st.subheader("1. The Challenge")
    st.info(f"Challenge: Write a prompt that forces the AI to be **simple** and **concise**. The output must have a Readability Score **above {target_flesch}** AND a Token Count **below {target_tokens}**.")
    
    prompt = st.text_area("Your Optimization Prompt:", "Define Artificial Intelligence.", height=100, key='h5_prompt')
    
    if 'h5_attempts' not in st.session_state:
        st.session_state.h5_attempts = []
    
    if st.button("Run Attempt (Step 2)", key='h5_run', type='primary'):
        if not prompt.strip():
            st.warning("Please enter a prompt.")
            return

        with st.spinner("Running optimization attempt..."):
            messages = [{"role": "user", "content": prompt}]
            # Use the target_tokens as max_tokens for the LLM call
            result = llm_call_cerebras(messages, max_tokens=target_tokens, model=DEFAULT_MODEL, temperature=0.3) 
            
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
            st.error(f"❌ Attempt {last_attempt['id']} Failed. Analyze the metrics below and adjust your prompt and click 'Run Attempt' again.")
        
        col_f, col_t = st.columns(2)
        col_f.metric(f"Current Readability (Target $\geq$ {target_flesch})", last_attempt['flesch_score'], 
                     delta_color='normal' if last_attempt['flesch_score'] >= target_flesch else 'inverse', delta="Needs to be simple!")
        col_t.metric(f"Current Tokens (Target $\leq$ {target_tokens})", last_attempt['tokens'],
                     delta_color='normal' if last_attempt['tokens'] <= target_tokens else 'inverse', delta="Needs to be short!")
        
        st.markdown("---")
        st.code(last_attempt['response'], language='markdown')
        
        df_attempts = pd.DataFrame(st.session_state.h5_attempts)
        fig = px.line(df_attempts, x='id', y=['flesch_score', 'tokens'], 
                      title="Performance Metrics Over Attempts",
                      labels={'id': 'Attempt Number', 'value': 'Score/Tokens'},
                      color_discrete_map={'flesch_score': '#0d47a1', 'tokens': '#B8860B'})
        st.plotly_chart(fig, use_container_width=True)

def render_learning_journal():
    st.header("📘 Learning Journal & Progress")
    st.markdown("##### **Goal:** Review and reflect on the key experiments you've run in each lab.")
    st.markdown("---")
    
    st.subheader("Your Module Progress")
    progress_percent = get_progress_percent()
    st.progress(progress_percent, text=f"Module Completion: **{progress_percent}%**")
    
    cols = st.columns(5)
    for i in range(1, 6):
        lab_key = f'H{i}'
        status = get_progress_badge(lab_key)
        cols[i-1].metric(f"Lab {i}", f"{lab_key} Status", status)
        
    st.markdown("---")
    
    st.subheader("Saved Experiments & Reflections")
    if st.session_state.journal:
        # Reverse the journal to show the newest entries first
        reversed_journal = st.session_state.journal[::-1]
        for entry in reversed_journal:
            with st.expander(f"**[{entry['timestamp'].split(' ')[0]}] {entry['lab']}: {entry['title']}**", expanded=False):
                st.markdown(f"**Prompt Used:**")
                st.code(entry['prompt'], language='markdown')
                st.markdown(f"**AI Response:**")
                st.info(entry['result'].get('content', 'N/A'))
                
                if 'reflection' in entry['metrics']:
                    st.markdown(f"**Your Reflection:** *{entry['metrics']['reflection']}*")
                
                if entry['metrics']:
                    st.markdown(f"**Metrics:** {entry['metrics']}")
    else:
        st.info("Your journal is empty! Start with the H1 lab to save your first experiment.")


# --- 5. AI ASSISTANT FUNCTION (LLM INTEGRATED - REFINED) ---

def render_ai_assistant_sidebar():
    """Renders the persistent AI Assistant and Progress Tracker in the sidebar."""
    
    # 1. Progress Tracker
    st.sidebar.markdown('<div class="progress-tracker">', unsafe_allow_html=True)
    st.sidebar.markdown("#### 🎯 Your Learning Progress")
    progress_percent = get_progress_percent()
    st.sidebar.progress(progress_percent, text=f"**Module Complete: {progress_percent}%**")
    
    lab_statuses = [f"**{k}** {v}" for k, v in st.session_state.progress.items()]
    st.sidebar.caption(f"Status: {' | '.join(lab_statuses)}")
    
    # 2. Guidance Message Display
    guidance_message = st.session_state.get('guidance', "Welcome! Select a lab tab (H1-H5) to begin your module.")
    st.sidebar.markdown('**Current Goal:**')
    st.sidebar.info(guidance_message)
    st.sidebar.markdown('</div>', unsafe_allow_html=True)

    # 3. AI Assistant Chat Interface
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 💬 AI Instructor Chatbot")
    st.sidebar.caption("Ask simple, non-technical questions here!")
    
    # Predefined System Context for the AI Assistant (Simplified)
    SYSTEM_PROMPT = """
    You are the **AI Instructor Assistant** for absolute beginners learning Prompt Engineering. 
    Your tone must be non-technical, extremely simple, and encouraging. 
    Your goal is to define concepts (like 'What is Temperature?'), suggest next steps, and give simple troubleshooting help.
    
    The user is currently working on labs H1 through H5 (Exploring Prompts, Roles, Temperature, Comparison, Optimization).
    
    **Example Responses (Use simple analogies):**
    - "What is a prompt?": "A prompt is like a **text message** you send to the AI telling it exactly what you want it to do."
    - "What is Temperature?": "Temperature is the **creativity dial**. Low temperature means predictable answers, high temperature means wild and creative answers!"
    - "What should I do now?": "You just finished H1! You should now click the **H2: Role Prompt Designer** tab to try giving the AI a personality."
    """
    
    user_query = st.sidebar.text_input("Ask about the module, steps, or concepts:", key="assistant_query")
    
    if st.sidebar.button("Ask Instructor", key="run_assistant"):
        
        # --- ASSISTANT COOLDOWN CHECK ---
        if time.time() - st.session_state.last_assistant_call < 5:
            st.sidebar.error("Please wait 5 seconds before asking the Assistant another question.")
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
                    model=DEFAULT_MODEL, 
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
                response = f"**Assistant Error:** Sorry, I encountered a connection issue. Please check your API key or try again in a minute."
            else:
                response = "I'm experiencing a service interruption. Please try again in a moment."

            # Add assistant message to history and display
            st.session_state.assistant_chat_history.append({"role": "assistant", "content": response})
            st.rerun() 

    # 4. Display Chat History
    for message in st.session_state.assistant_chat_history:
        if message['role'] == 'user':
            st.sidebar.markdown(f'<div class="user-message">**You:** {message["content"]}</div>', unsafe_allow_html=True)
        elif message['role'] == 'assistant':
            st.sidebar.markdown(f'<div class="assistant-message">**Instructor:** {message["content"]}</div>', unsafe_allow_html=True)

    
    # 5. Reset Button
    st.sidebar.markdown("---")
    if st.sidebar.button("Reset All Lab Progress ⚠️", type='secondary'):
        # Clear all session state keys
        for key in list(st.session_state.keys()):
             del st.session_state[key]

        # Initialize base state again
        st.session_state.progress = {f'H{i}': '🔴' for i in range(1, 6)} 
        st.session_state.journal = []
        st.session_state.lab_results = {}
        st.session_state.current_tab = 'Intro' # Reset to Intro
        st.session_state.guidance = "Welcome! Select a lab tab (H1-H5) to begin your module." 
        st.session_state.h1_step = 1 
        st.session_state.h4_results = pd.DataFrame() 
        st.session_state.onboarding_done = False
        st.session_state.assistant_chat_history = [{"role": "assistant", "content": "👋 Hi there! I'm your AI Instructor. Ask me anything in simple terms about prompts, AI settings (like 'Temperature'), or what to do next!"}]
        st.session_state.last_assistant_call = 0 
        
        st.success("Session cleared. Please refresh the browser.")
        st.rerun()


# --- 6. MAIN APPLICATION ENTRY POINT ---

def render_main_page():
    
    # 3. Onboarding Modal (Must be called early)
    show_onboarding_modal()
    
    # Final App Title (Enhanced)
    st.markdown('<div class="title-header">Module 1: AI + Prompt Engineering Foundations</div>', unsafe_allow_html=True)
    st.markdown("---")

    # Tab Titles - Added Getting Started Tab
    tab_titles = [
        "🧭 Getting Started", "H1: First Prompt 🚀", "H2: Role Designer 🎭", 
        "H3: Temp & Context 🌡️", "H4: Comparison ⭐", 
        "H5: Optimization 🎯", "📘 Learning Journal"
    ]
    tabs = st.tabs(tab_titles)
    
    # Content Rendering
    with tabs[0]:
        st.session_state.current_tab = 'Intro'
        render_getting_started()
    with tabs[1]:
        st.session_state.current_tab = 'H1'
        render_lab1()
    with tabs[2]:
        st.session_state.current_tab = 'H2'
        render_lab2()
    with tabs[3]:
        st.session_state.current_tab = 'H3'
        render_lab3()
    with tabs[4]:
        st.session_state.current_tab = 'H4'
        render_lab4()
    with tabs[5]:
        st.session_state.current_tab = 'H5'
        render_lab5()
    with tabs[6]:
        st.session_state.current_tab = 'Journal'
        render_learning_journal()

if __name__ == '__main__':
    render_ai_assistant_sidebar()
    render_main_page()
