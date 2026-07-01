import gradio as gr
import subprocess
import pandas as pd
import os

# -- Monkeypatch for Gradio API Schema Bug --
import gradio_client.utils
_original_get_type = gradio_client.utils.get_type
def safe_get_type(schema):
    if isinstance(schema, bool):
        return "Any"
    return _original_get_type(schema)
gradio_client.utils.get_type = safe_get_type
# -------------------------------------------


def run_ranking():
    try:
        # Run the rank.py script via subprocess
        result = subprocess.run(
            ["python", "rank.py", "--candidates", "data/jobs.json", "--out", "submission.csv"],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Load and display the CSV if successful
        if os.path.exists("submission.csv"):
            df = pd.read_csv("submission.csv")
            return "Ranking completed successfully!\n\n" + result.stdout, df, "submission.csv"
        else:
            return "Error: submission.csv was not generated.\n\n" + result.stdout, None, None
            
    except subprocess.CalledProcessError as e:
        return f"An error occurred while running the ranking system:\n\n{e.stderr}", None, None

# Build the Gradio Interface
with gr.Blocks(title="Redrop AI Sandbox") as demo:
    gr.Markdown("# Redrop AI - Candidate Ranking Sandbox")
    gr.Markdown("""
    This is a live sandbox environment for the Redrop AI V6 candidate ranking system. 
    Click the button below to run the deterministic ranking pipeline on the pre-loaded sample candidates.
    """)
    
    run_btn = gr.Button("Run Ranking System", variant="primary")
    
    with gr.Row():
        logs_output = gr.Textbox(label="Execution Logs", lines=10)
        
    with gr.Row():
        table_output = gr.Dataframe(label="Ranked Candidates (submission.csv)")
        
    with gr.Row():
        file_output = gr.File(label="Download submission.csv")
        
    # Wire the button to the function
    run_btn.click(
        fn=run_ranking, 
        inputs=[], 
        outputs=[logs_output, table_output, file_output]
    )

if __name__ == "__main__":
    demo.launch()
