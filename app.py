import subprocess
import pandas as pd
import os

# Monkeypatch BEFORE importing gradio so the fix is in place at load time
import gradio_client.utils as _gcu
_orig = _gcu.get_type
def _safe_get_type(schema):
    if isinstance(schema, bool):
        return "Any"
    return _orig(schema)
_gcu.get_type = _safe_get_type

import gradio as gr


def run_ranking():
    try:
        result = subprocess.run(
            ["python", "rank.py", "--candidates", "data/jobs.json", "--out", "submission.csv"],
            capture_output=True,
            text=True,
            timeout=300
        )
        logs = result.stdout + ("\n" + result.stderr if result.stderr else "")

        if os.path.exists("submission.csv"):
            df = pd.read_csv("submission.csv")
            preview = df.head(20).to_string(index=False)
            return f"✅ Ranking completed!\n\n{logs}\n\nTop 20 results preview:\n{preview}"
        else:
            return f"❌ submission.csv was not produced.\n\n{logs}"

    except subprocess.TimeoutExpired:
        return "❌ Timed out after 5 minutes."
    except Exception as e:
        return f"❌ Error: {e}"


with gr.Blocks(title="Redrop AI Sandbox") as demo:
    gr.Markdown("# 🚀 Redrop AI — Candidate Ranking Sandbox")
    gr.Markdown(
        "Live sandbox for the **Redrop AI V6** deterministic ranking pipeline. "
        "Click the button to run end-to-end ranking on the pre-loaded sample candidates "
        "and download the ranked CSV."
    )

    run_btn = gr.Button("▶ Run Ranking System", variant="primary", size="lg")
    output_box = gr.Textbox(label="Output / Logs", lines=20, interactive=False)
    run_btn.click(fn=run_ranking, inputs=[], outputs=[output_box])

    gr.Markdown("---")
    gr.Markdown("📄 After running, download `submission.csv` directly from the **Files** tab above.")


demo.launch()
