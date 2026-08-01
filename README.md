
# PromptForge — setup

1. `pip install -r requirements.txt`
2. Copy `.env.example` to `.env` and fill in whichever API key you'll use (Claude or OpenAI — you don't need both).
3. `streamlit run app.py`
4. Type your rough idea, expand "Add context" if you want a sharper result, hit Generate.

Everything gets logged to `prompt_history.json` in the same folder so you can go back to old prompts.
