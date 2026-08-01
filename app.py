"""
PromptForge — turns a rough idea into an optimized LLM prompt.
Run: streamlit run app.py
"""

import json
import os
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from prompt_engine import generate_prompt, refine_prompt, test_run_prompt, check_prompt_quality

load_dotenv()

HISTORY_FILE = "prompt_history.json"

st.set_page_config(page_title="PromptForge", page_icon="🛠️", layout="centered")

# ---------- presets ----------
PRESETS = {
    "CogniTrack coding prompt": {
        "task_type": "Coding",
        "target_llm": "Claude",
        "output_format": "Complete final code file, not a diff",
        "tone": "Direct, concise",
        "constraints": "No f-strings, no list comprehensions, no enumerate(), no unnecessary imports. Simple explicit code.",
        "extra_context": "Part of CogniTrack — a Streamlit + FastAPI app integrating EEG stress classification, Arduino pulse sensor, Whisper transcription, Gemini AI copilot, and MediaPipe facial analysis.",
        "include_examples": False,
    },
    "Exam-oriented explanation": {
        "task_type": "Explanation/Learning",
        "target_llm": "Any",
        "output_format": "Structured, exam-answer style with headings and short points",
        "tone": "Simple, exam-oriented, no fluff",
        "constraints": "Keep code examples simple and explicit, no advanced syntax",
        "extra_context": "",
        "include_examples": True,
    },
    "Debug my code": {
        "task_type": "Debugging",
        "target_llm": "Any",
        "output_format": "Explain the bug first, then give the corrected full code",
        "tone": "Direct",
        "constraints": "Don't rewrite unrelated parts of the code",
        "extra_context": "",
        "include_examples": False,
    },
}


def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []


def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


for key, default in {
    "history": load_history(),
    "last_result": None,
    "task_type": "",
    "target_llm": "",
    "output_format": "",
    "tone": "",
    "constraints": "",
    "extra_context": "",
    "include_examples": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

st.title("🛠️ PromptForge")
st.caption("Give it your rough idea. It hands back the prompt you should actually use.")

# ---------- sidebar ----------
with st.sidebar:
    st.header("Settings")
    provider = st.selectbox("LLM to generate the prompt", ["claude", "openai", "gemini"])

    if provider == "claude":
        api_key = st.text_input("Anthropic API key", type="password", value=os.getenv("ANTHROPIC_API_KEY", ""))
        model = st.text_input("Model", value="claude-sonnet-4-6")
    elif provider == "openai":
        api_key = st.text_input("OpenAI API key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
        model = st.text_input("Model", value="gpt-4o")
    else:
        api_key = st.text_input("Google API key", type="password", value=os.getenv("GOOGLE_API_KEY", ""))
        model = st.text_input("Model", value="gemini-3.5-flash")

    st.divider()
    st.subheader("Presets")
    for name in PRESETS:
        if st.button(name, use_container_width=True):
            for k, v in PRESETS[name].items():
                st.session_state[k] = v
            st.rerun()

    st.divider()
    if st.button("Clear history"):
        st.session_state.history = []
        save_history([])
        st.rerun()

# ---------- main input ----------
raw_query = st.text_area(
    "What do you actually want the LLM to do?",
    placeholder="e.g. help me write a report on EEG stress classification for my project",
    height=100,
)

with st.expander("Add context (optional, but improves the prompt a lot)"):
    col1, col2 = st.columns(2)
    with col1:
        task_type = st.selectbox(
            "Task type",
            ["", "Coding", "Writing", "Research/Analysis", "Debugging", "Explanation/Learning",
             "Data analysis", "Design/Creative", "Other"],
            key="task_type",
        )
        target_llm = st.selectbox("Target LLM this prompt is for", ["", "Claude", "GPT", "Gemini", "Any"], key="target_llm")
    with col2:
        output_format = st.text_input("Desired output format", placeholder="e.g. Python code, bullet points, table", key="output_format")
        tone = st.text_input("Tone", placeholder="e.g. concise, exam-oriented, formal", key="tone")

    constraints = st.text_input("Constraints / things to avoid", placeholder="e.g. no f-strings, keep under 300 words", key="constraints")
    extra_context = st.text_area("Extra background", placeholder="Anything else the LLM should know", height=68, key="extra_context")
    include_examples = st.checkbox("Ask it to include examples in the generated prompt", key="include_examples")

generate_btn = st.button("Generate perfect prompt", type="primary", use_container_width=True)

if generate_btn:
    if not raw_query.strip():
        st.error("Type your rough idea first.")
    elif not api_key:
        st.error("Add your API key in the sidebar.")
    else:
        context = {
            "task_type": task_type,
            "target_llm": target_llm,
            "output_format": output_format,
            "tone": tone,
            "constraints": constraints,
            "extra_context": extra_context,
            "include_examples": include_examples,
        }
        with st.spinner("Forging your prompt..."):
            try:
                result = generate_prompt(provider, raw_query, context, api_key, model)
                st.session_state.last_result = result

                entry = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "raw_query": raw_query,
                    "context": context,
                    "generated_prompt": result,
                    "provider": provider,
                    "starred": False,
                }
                st.session_state.history.insert(0, entry)
                save_history(st.session_state.history)
            except Exception as e:
                st.error(f"Something went wrong: {e}")

# ---------- result + refine + test-run + quality check ----------
if st.session_state.last_result:
    st.subheader("Your optimized prompt")
    st.code(st.session_state.last_result, language="markdown")

    st.download_button(
        "Download as .txt",
        st.session_state.last_result,
        file_name="prompt.txt",
        use_container_width=True,
    )

    # quality checklist
    with st.expander("Quality checklist"):
        checks = check_prompt_quality(st.session_state.last_result)
        for label, passed in checks.items():
            st.write(("✅ " if passed else "⬜ ") + label)

    # refine loop
    st.markdown("**Not quite right? Tweak it:**")
    refine_col1, refine_col2 = st.columns([3, 1])
    with refine_col1:
        refine_instruction = st.text_input(
            "Refinement instruction", placeholder="e.g. make it shorter, add more constraints",
            label_visibility="collapsed",
        )
    with refine_col2:
        refine_btn = st.button("Apply", use_container_width=True)

    if refine_btn:
        if not refine_instruction.strip():
            st.error("Type what you want changed.")
        else:
            with st.spinner("Refining..."):
                try:
                    refined = refine_prompt(provider, st.session_state.last_result, refine_instruction, api_key, model)
                    st.session_state.last_result = refined
                    st.rerun()
                except Exception as e:
                    st.error(f"Something went wrong: {e}")

    # test run
    if st.button("Test-run this prompt (see what it produces)", use_container_width=True):
        with st.spinner("Running..."):
            try:
                output = test_run_prompt(provider, st.session_state.last_result, api_key, model)
                st.markdown("**Output preview:**")
                st.info(output)
            except Exception as e:
                st.error(f"Something went wrong: {e}")

# ---------- history: search + favorites ----------
if st.session_state.history:
    st.divider()
    st.subheader("History")

    search_term = st.text_input("Search history", placeholder="search by keyword")
    show_starred_only = st.checkbox("Show favorites only")

    filtered = st.session_state.history
    if search_term:
        term = search_term.lower()
        filtered = [
            e for e in filtered
            if term in e["raw_query"].lower() or term in e["generated_prompt"].lower()
        ]
    if show_starred_only:
        filtered = [e for e in filtered if e.get("starred")]

    for i, entry in enumerate(filtered[:20]):
        star = "⭐" if entry.get("starred") else "☆"
        with st.expander(f"{star} {entry['timestamp']} — {entry['raw_query'][:60]}"):
            st.write("**Raw query:**", entry["raw_query"])
            st.code(entry["generated_prompt"], language="markdown")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Toggle favorite", key=f"star_{entry['timestamp']}_{i}"):
                    idx = st.session_state.history.index(entry)
                    st.session_state.history[idx]["starred"] = not st.session_state.history[idx].get("starred")
                    save_history(st.session_state.history)
                    st.rerun()
            with col2:
                if st.button("Load into editor", key=f"load_{entry['timestamp']}_{i}"):
                    st.session_state.last_result = entry["generated_prompt"]
                    st.rerun()