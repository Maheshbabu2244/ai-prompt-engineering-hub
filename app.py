import streamlit as st
import google.generativeai as genai
import time
import random
import pandas as pd
import sqlite3
import json
import base64

# --- Page Configuration ---
st.set_page_config(
    page_title="AI+ Prompt Engineer Dashboard",
    page_icon="🚀",
    layout="wide"
)

# --- "Nexus" Theme: Custom CSS (Final Version) ---
def add_custom_css():
    @st.cache_data()
    def get_base64_bg():
        svg = """<svg xmlns='http://www.w3.org/2000/svg' width='100' height='100' viewBox='0 0 100 100'><defs><linearGradient id='g' x1='0%' y1='0%' x2='100%' y2='100%'><stop offset='0%' stop-color='#00ffff' stop-opacity='0.1'/><stop offset='100%' stop-color='#00aaff' stop-opacity='0.1'/></linearGradient></defs><circle cx='50' cy='50' r='2' fill='#00aaff' fill-opacity='0.3'/><g fill='none' stroke-width='0.5' stroke='url(#g)'><path d='M50 50 L0 0 M50 50 L100 0 M50 50 L0 100 M50 50 L100 100 M50 50 L50 0 M50 50 L50 100 M50 50 L0 50 M50 50 L100 50'/></g></svg>"""
        return base64.b64encode(svg.encode()).decode()

    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
        
        .stApp, .stApp h1, .stApp h2, .stApp h3, .stApp p, .stApp li, .stApp small, .stApp button, .stApp textarea, .stApp input {{
            font-family: 'Roboto', sans-serif !important;
        }}
        .stApp {{ background-color: #0a0a1a; background-image: url("data:image/svg+xml;base64,{get_base64_bg()}"); background-size: 300px 300px; color: #EAEAEA; }}
        h1, h2, h3 {{ color: #FFFFFF; }}

        .kpi-card {{
            background-color: rgba(20, 20, 40, 0.7); border-radius: 12px; padding: 25px;
            box-shadow: 0 0 15px rgba(0, 255, 255, 0.1); text-align: center;
            border: 1px solid rgba(0, 255, 255, 0.2); backdrop-filter: blur(5px); transition: all 0.3s ease;
        }}
        .kpi-card:hover {{ border-color: rgba(0, 255, 255, 0.5); transform: translateY(-5px); }}
        .kpi-card h3 {{ font-size: 18px; color: #CCCCCC; margin-bottom: 12px; font-weight: 300; }}
        .kpi-card p {{ font-size: 42px; font-weight: 700; color: #00FFFF; margin: 0; }}
        .kpi-card small {{ font-size: 14px; color: #8899AA; }}

        [data-testid="stExpander"] {{ border: 1px solid rgba(0, 255, 255, 0.2); border-radius: 10px; background-color: rgba(10, 10, 30, 0.8); }}
        [data-testid="stExpander"] summary p {{ font-size: 1.1rem !important; font-weight: bold !important; color: #EAEAEA !important; }}

        [data-testid="stSidebar"] {{ background-color: #0f0f20; border-right: 1px solid rgba(0, 255, 255, 0.2); }}
        
        .stButton>button {{ background-color: #00FFFF; color: #0a0a1a; border-radius: 8px; font-weight: bold; border: none; }}
        .stButton>button:hover {{ background-color: #FFFFFF; color: #0a0a1a; }}
        
        .vitals-bar {{ display: flex; justify-content: space-around; background-color: rgba(20, 20, 40, 0.7); border-radius: 10px; padding: 10px; margin-bottom: 20px; border: 1px solid rgba(0, 255, 255, 0.2); }}
        .vital {{ text-align: center; }}
        .vital-title {{ font-size: 14px; color: #8899AA; }}
        .vital-value {{ font-size: 18px; font-weight: bold; color: #00FFFF; }}
        .proceedings-log {{ background-color: rgba(10,10,30,0.8); border: 1px solid rgba(0, 255, 255, 0.2); padding: 15px; border-radius: 10px; height: 150px; overflow-y: auto; font-family: 'Courier New', Courier, monospace; font-size: 14px; }}
    </style>
    """, unsafe_allow_html=True)

add_custom_css()

# --- API Key & Session State Initialization ---
def initialize_app():
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    except KeyError:
        st.error("Gemini API key is not set. Please create a .streamlit/secrets.toml file and add it.", icon="🚨")
        st.stop()
    if 'prompts_analyzed' not in st.session_state:
        st.session_state.prompts_analyzed = 1741; st.session_state.techniques_mastered = 4
        st.session_state.avg_quality_score = 91.6; st.session_state.active_projects = 3
        st.session_state.realtime_active = True; st.session_state.proceedings = ["System Initialized... AI Engine Active."]
initialize_app()

# --- Database for Module 6 ---
@st.cache_resource
def init_db():
    conn = sqlite3.connect('module6_projects.db', check_same_thread=False)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS projects (id INTEGER PRIMARY KEY, title TEXT, description TEXT, prompt_strategy TEXT, evaluation_notes TEXT)''')
    conn.commit()
    return conn
conn = init_db()

# --- Gemini Helper Function ---
@st.cache_data(ttl=3600)
def generate_gemini_response(prompt, model_name="gemini-2.5-flash", is_json=False):
    try:
        model = genai.GenerativeModel(model_name)
        config = genai.types.GenerationConfig(response_mime_type="application/json" if is_json else "text/plain")
        response = model.generate_content(prompt, generation_config=config)
        return response.text
    except Exception as e:
        return f"Gemini Error: {str(e)}"

# --- Module Content Data ---
MODULE_CONTENT = {1: {"name": "Module 1: Foundations of AI", "problem": "To engineer effective prompts, one must first understand the context and capabilities of AI models, which are rooted in their history.", "objective": "Understand the significance of key milestones in AI history and their impact on modern generative models.", "usage": "1. **Select Milestone:** Choose an event.\n2. **Generate Explanation:** Click the button.\n3. **Review AI Report:** Read the Gemini-generated summary."}, 2: {"name": "Module 2: Principles of Effective Prompting", "problem": "Vague or unstructured instructions lead to generic and unhelpful AI responses.", "objective": "Master how to give clear direction by defining a persona and specifying a response format to improve AI output quality.", "usage": "1. **Define Persona:** Describe the AI's role.\n2. **Specify Format:** Choose the output structure.\n3. **Enter Task:** Write the command.\n4. **Generate Response:** See how the AI combines all elements."}, 3: {"name": "Module 3: AI Tools and Models", "problem": "Different AI models are optimized for different tasks (e.g., speed vs. power). Choosing the right one is critical.", "objective": "Observe the qualitative differences in output between a fast model (Gemini 1.5 Flash) and a powerful one (Gemini 1.5 Pro).", "usage": "1. **Enter Prompt:** Write a prompt for both models.\n2. **Compare Models:** Generate responses side-by-side.\n3. **Analyze Outputs:** Note the differences in detail, nuance, and style."}, 4: {"name": "Module 4: Mastering Prompting Techniques", "problem": "Simple prompts are often insufficient for solving complex, multi-step problems.", "objective": "Learn to use an AI to structure prompts for advanced techniques like Chain-of-Thought (CoT) to guide another AI through complex reasoning.", "usage": "1. **Select Technique:** Choose CoT or RAG.\n2. **Describe Task:** Enter a complex problem.\n3. **Generate Advanced Prompt:** Let Gemini create a structured 'meta-prompt' to solve the task."}, 5: {"name": "Module 5: Mastering Image Model Techniques", "problem": "Basic text descriptions lead to generic or stylistically plain AI-generated images.", "objective": "Master enhancing simple ideas with style modifiers and quality boosters to generate detailed, artistic prompts for text-to-image models.", "usage": "1. **Enter Idea:** Type a simple image concept.\n2. **Choose Style:** Select an artistic style.\n3. **Enhance Prompt:** Let Gemini combine these into a rich, descriptive paragraph."}, 6: {"name": "Module 6: Project-Based Learning", "problem": "Theoretical knowledge needs to be applied to structured, real-world scenarios to be retained.", "objective": "Gain practical experience by defining, executing, and evaluating a complete AI project from start to finish.", "usage": "1. **Define Project:** Fill out the form with your plan.\n2. **Add Project:** Save your plan to the local database.\n3. **Review & Manage:** View all saved projects below."}, 7: {"name": "Module 7: Ethical Considerations", "problem": "AI models can inherit and amplify human biases from their training data, leading to unfair or harmful outcomes.", "objective": "Develop the critical ability to analyze prompts for potential ethical risks, ensuring the responsible use of AI.", "usage": "1. **Enter Prompt:** Type a prompt with potential hidden biases.\n2. **Analyze for Bias:** Let Gemini act as an AI ethics expert.\n3. **Review Report:** Examine the AI-generated JSON report on the risk level."}}

# --- PAGE DEFINITIONS ---

def dashboard_page():
    st.title("AI+ Prompt Engineering Dashboard")
    st.markdown("<h3>System Vitals</h3>", unsafe_allow_html=True)
    st.markdown("""<div class="vitals-bar"><div class="vital"><div class="vital-title">AI Engine Status</div><div class="vital-value" style="color: #00FF7F;">● ACTIVE</div></div><div class="vital"><div class="vital-title">System Integrity</div><div class="vital-value">100%</div></div><div class="vital"><div class="vital-title">Current Load</div><div class="vital-value">LOW</div></div></div>""", unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1]); col1.subheader("Real-time Learner Analytics"); col2.toggle("Live Analytics", value=st.session_state.realtime_active, key="realtime_active")
    kpi_cols = st.columns(4)
    kpi_cols[0].markdown(f'<div class="kpi-card"><h3>Prompts Analyzed</h3><p>{st.session_state.prompts_analyzed}</p><small>Total in labs</small></div>', unsafe_allow_html=True)
    kpi_cols[1].markdown(f'<div class="kpi-card"><h3>Techniques Mastered</h3><p>{st.session_state.techniques_mastered}/7</p><small>Core methods</small></div>', unsafe_allow_html=True)
    kpi_cols[2].markdown(f'<div class="kpi-card"><h3>Avg. Quality Score</h3><p>{st.session_state.avg_quality_score:.1f}%</p><small>Lab evaluations</small></div>', unsafe_allow_html=True)
    kpi_cols[3].markdown(f'<div class="kpi-card"><h3>Active Projects</h3><p>{st.session_state.active_projects}</p><small>In Project Hub</small></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True); st.subheader("Live Proceedings Log")
    log_html = "<div class='proceedings-log'>" + "<br>".join(st.session_state.proceedings) + "</div>"
    st.markdown(log_html, unsafe_allow_html=True)

def module_labs_page(module_number):
    content = MODULE_CONTENT[module_number]
    st.title(content["name"])
    with st.expander("How to Use This Lab", expanded=True):
        st.markdown(f"**Problem Statement:** {content['problem']}"); st.markdown(f"**Learning Objective:** {content['objective']}"); st.markdown("---"); st.subheader("Step-by-Step Usage:"); st.markdown(content["usage"])
    st.header("Interactive Lab"); st.markdown("---")

    if module_number == 1:
        milestone = st.selectbox("Select a milestone:", ["The Turing Test", "The Dartmouth Workshop", "Deep Blue vs. Garry Kasparov", "The rise of Transformers"])
        if st.button("Explain Milestone", type="primary"):
            with st.spinner("Generating..."): st.success(generate_gemini_response(f"Explain the significance of '{milestone}'."))
    
    elif module_number == 2:
        persona = st.text_area("1. Define AI Persona:", "You are a witty, enthusiastic tour guide for Bengaluru.")
        format_spec = st.selectbox("2. Specify Response Format:", ["Bulleted list", "JSON object", "Short paragraph"])
        task = st.text_input("3. Enter Task:", "What are three must-visit places in Bengaluru?")
        if st.button("Generate Response", type="primary"):
            with st.spinner("Generating..."):
                prompt = f"**Persona:** {persona}\n\n**Task:** {task}\n\n**Format:** {format_spec}."
                st.markdown(generate_gemini_response(prompt, is_json="JSON" in format_spec))
    
    elif module_number == 3:
        prompt = st.text_area("Enter a prompt to compare models:", "Explain quantum computing in a simple analogy.")
        if st.button("Compare Models", type="primary"):
            col1, col2 = st.columns(2)
            with col1:
                st.info("#### Gemini 1.5 Flash (Fast)"); 
                with st.spinner("Generating..."): st.markdown(generate_gemini_response(prompt, "gemini-2.5-flash"))
            with col2:
                st.success("#### Gemini 1.5 Pro (Powerful)")
                with st.spinner("Generating..."): st.markdown(generate_gemini_response(prompt, "gemini-2.5-pro"))

    elif module_number == 4:
        technique = st.selectbox("Select a technique:", ["Chain-of-Thought", "Retrieval Augmented Generation"])
        task = st.text_area("Describe the complex task:", "Plan a budget-friendly 7-day trip to Italy from Bengaluru.")
        if st.button("Generate Advanced Prompt", type="primary"):
            with st.spinner("Generating..."):
                sys_prompt = f"Create a perfect '{technique}' prompt for another AI to solve this task: {task}"
                st.code(generate_gemini_response(sys_prompt, "gemini-2.5-pro"), language="markdown")
            
    elif module_number == 5:
        idea = st.text_input("1. Simple image idea:", "A robot reading a book in a library.")
        style = st.selectbox("2. Artistic Style:", ["Photorealistic", "Digital Art", "Cyberpunk"])
        if st.button("Enhance Prompt", type="primary"):
            with st.spinner("Generating..."):
                sys_prompt = f"Expand this idea into a breathtaking image prompt. Idea: '{idea}'. Style: '{style}'."
                st.code(generate_gemini_response(sys_prompt, "gemini-2.5-pro"), language="text")

    elif module_number == 6:
        with st.form("project_form"):
            title, desc, strategy, eval_notes = st.text_input("Title"), st.text_area("Description"), st.text_area("Strategy"), st.text_area("Evaluation")
            if st.form_submit_button("Add Project"):
                c = conn.cursor(); c.execute("INSERT INTO projects (title, description, prompt_strategy, evaluation_notes) VALUES (?, ?, ?, ?)", (title, desc, strategy, eval_notes)); conn.commit(); st.success("Project added!")
        st.subheader("Existing Projects"); df = pd.read_sql_query("SELECT id, title FROM projects", conn); st.dataframe(df, use_container_width=True)

    elif module_number == 7:
        prompt = st.text_area("Enter a prompt to analyze for bias:", "List suitable candidates for a 'construction manager' position.")
        if st.button("Analyze for Bias", type="primary"):
            with st.spinner("Generating..."):
                sys_prompt = f"Analyze the prompt '{prompt}' for ethical biases. Respond as a JSON with 'risk_level' and 'justification'."
                st.json(generate_gemini_response(sys_prompt, "gemini-2.5-pro", is_json=True))

def advanced_tools_page(tool_name):
    st.title(f"Advanced AI Toolkit: {tool_name}")
    if tool_name == "Advanced Prompt Studio":
        st.header("Optimize and Analyze Your Prompts")
        prompt = st.text_area("1. Enter basic prompt:", height=100)
        col1, col2, col3 = st.columns(3); persona = col1.selectbox("Persona", ["Expert"]); tone = col2.selectbox("Tone", ["Formal"]); format_spec = col3.selectbox("Format", ["JSON"])
        if st.button("Optimize & Analyze", type="primary"):
            sys_prompt = f'Rewrite this prompt: "{prompt}" with Persona: {persona}, Tone: {tone}, Format: {format_spec}. Then, critique the original. Respond as a JSON with "optimized_prompt" and "critique".'
            response = generate_gemini_response(sys_prompt, "gemini-2.5-pro", is_json=True)
            try:
                data = json.loads(response)
                st.subheader("Optimized Prompt"); st.code(data['optimized_prompt'], language="markdown")
                st.subheader("Critique"); st.info(data['critique'])
            except (json.JSONDecodeError, KeyError): st.error("Failed to get valid analysis. Raw output: " + response)
    elif tool_name == "Chain-of-Thought (CoT) Visualizer":
        st.header("Visualize an AI's Reasoning Process")
        problem = st.text_area("Enter logic puzzle:", height=120)
        if st.button("Visualize Reasoning", type="primary"):
            sys_prompt = f"Solve this step-by-step. Start each step with 'Step X:'. Problem: {problem}"
            reasoning = generate_gemini_response(sys_prompt, "gemini-2.5-pro")
            st.subheader("AI's Reasoning Path"); steps = reasoning.split("Step ")[1:]
            for i, step_text in enumerate(steps):
                with st.container(border=True): st.markdown(f"**Step {step_text.strip()}")
                if i < len(steps) - 1: st.markdown("↓", unsafe_allow_html=True)

# --- SIDEBAR & NAVIGATION LOGIC ---
st.sidebar.title("Selection Menu")
st.sidebar.markdown("---")
page_selection = st.sidebar.radio("Navigation", ["Dashboard", "Course Module Labs", "Advanced AI Toolkit"], label_visibility="hidden")

if page_selection == "Dashboard":
    dashboard_page()
    if st.session_state.realtime_active:
        if random.random() < 0.3:
            proceedings = ["New technique explored.", "Project updated.", "Ethical analysis complete."]
            st.session_state.proceedings.insert(0, f"LOG: {random.choice(proceedings)}")
            if len(st.session_state.proceedings) > 5: st.session_state.proceedings.pop()
        st.session_state.prompts_analyzed += random.randint(1, 5)
        st.session_state.avg_quality_score = max(80, min(100, st.session_state.avg_quality_score + random.uniform(-0.1, 0.1)))
        time.sleep(2); st.rerun()
elif page_selection == "Course Module Labs":
    st.sidebar.markdown("---")
    module_names = [MODULE_CONTENT[i]['name'] for i in range(1, 8)]
    module_selection = st.sidebar.selectbox("Select a Module Lab", module_names)
    module_number = int(module_selection.split(":")[0].split(" ")[1])
    module_labs_page(module_number)
elif page_selection == "Advanced AI Toolkit":
    st.sidebar.markdown("---")
    tool_selection = st.sidebar.selectbox("Select a Tool", ["Advanced Prompt Studio", "Chain-of-Thought (CoT) Visualizer"])
    advanced_tools_page(tool_selection)