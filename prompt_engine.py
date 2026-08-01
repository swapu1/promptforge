"""
prompt_engine.py
Handles: building the meta-prompt, calling the chosen LLM (Claude/OpenAI/Gemini),
refining an existing generated prompt, and test-running a prompt to preview its output.
"""

import anthropic
import openai
import google.generativeai as genai


META_SYSTEM_PROMPT = """You are an expert prompt engineer. Your only job is to take a user's
raw request (plus any context they give you) and rewrite it into a single, highly effective
prompt that they can paste into an LLM to get the best possible result.

Rules for the prompt you generate:
1. Give the LLM a clear role if it helps (e.g. "You are a senior Python developer...").
2. State the exact task unambiguously.
3. Include all relevant context the user gave you.
4. Specify the desired output format (length, structure, code vs prose, etc).
5. Add constraints or things to avoid, if the user mentioned any.
6. If examples would help the target LLM understand the task, include 1-2 short ones.
7. If the task is complex, ask the LLM to think step by step before answering.

Output ONLY the final prompt text. No preamble, no explanation, no markdown fences,
no "Here is your prompt:" — just the prompt itself, ready to copy-paste.
"""

REFINE_SYSTEM_PROMPT = """You refine existing LLM prompts based on user feedback.
You will be given an original prompt and an instruction for how to change it.
Output ONLY the revised prompt text — no preamble, no explanation, no markdown fences."""


def build_user_message(raw_query: str, context: dict) -> str:
    """Packs the raw query + structured context into one message for the meta-prompt LLM."""
    parts = [f"Raw request from user: {raw_query}"]

    if context.get("task_type"):
        parts.append(f"Task type: {context['task_type']}")
    if context.get("target_llm"):
        parts.append(f"This prompt will be used on: {context['target_llm']}")
    if context.get("output_format"):
        parts.append(f"Desired output format: {context['output_format']}")
    if context.get("tone"):
        parts.append(f"Desired tone: {context['tone']}")
    if context.get("constraints"):
        parts.append(f"Constraints / things to avoid: {context['constraints']}")
    if context.get("include_examples"):
        parts.append("Include 1-2 short examples in the generated prompt if useful.")
    if context.get("extra_context"):
        parts.append(f"Extra background: {context['extra_context']}")

    return "\n".join(parts)


# ---------- generic call layer ----------

def call_llm(provider: str, system: str, user_message: str, api_key: str, model: str) -> str:
    """One function to call any of the three providers with a system + single user message."""
    if provider == "claude":
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user_message}],
        )
        return response.content[0].text.strip()

    elif provider == "openai":
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_message},
            ],
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()

    elif provider == "gemini":
        genai.configure(api_key=api_key)
        gen_model = genai.GenerativeModel(model_name=model, system_instruction=system)
        response = gen_model.generate_content(user_message)
        return response.text.strip()

    else:
        raise ValueError(f"Unknown provider: {provider}")


# ---------- feature-specific wrappers ----------

def generate_prompt(provider: str, raw_query: str, context: dict, api_key: str, model: str) -> str:
    """Turns a raw idea + context into an optimized prompt."""
    user_msg = build_user_message(raw_query, context)
    return call_llm(provider, META_SYSTEM_PROMPT, user_msg, api_key, model)


def refine_prompt(provider: str, original_prompt: str, instruction: str, api_key: str, model: str) -> str:
    """Takes an already-generated prompt and a tweak instruction, returns the revised prompt."""
    user_msg = f"Original prompt:\n{original_prompt}\n\nRefinement instruction: {instruction}"
    return call_llm(provider, REFINE_SYSTEM_PROMPT, user_msg, api_key, model)


def test_run_prompt(provider: str, prompt_text: str, api_key: str, model: str) -> str:
    """Actually runs the generated prompt as-is, so the user can preview the output it produces."""
    return call_llm(provider, "", prompt_text, api_key, model)


def check_prompt_quality(prompt_text: str) -> dict:
    """Lightweight heuristic checklist — flags what a well-formed prompt usually has."""
    lower = prompt_text.lower()
    checks = {
        "Has a role/persona": any(kw in lower for kw in ["you are", "act as", "as a "]),
        "Specifies output format": any(
            kw in lower for kw in ["format", "output", "respond with", "list", "bullet", "table", "code block", "structure"]
        ),
        "Has constraints": any(
            kw in lower for kw in ["avoid", "do not", "don't", "must not", "without", "constraint", "limit"]
        ),
        "Includes examples": any(kw in lower for kw in ["example", "e.g.", "for instance"]),
        "Asks for step-by-step reasoning": any(
            kw in lower for kw in ["step by step", "step-by-step", "think through", "first,", "reasoning"]
        ),
        "Reasonable length (not too short)": len(prompt_text.split()) >= 25,
    }
    return checks