import streamlit as st
import subprocess
import pandas as pd
import os

st.title("Redrop AI - Candidate Ranking Sandbox")

st.markdown("""
This is a live sandbox environment for the Redrop AI V6 candidate ranking system. 
Click the button below to run the deterministic ranking pipeline on the pre-loaded sample candidates.
""")

if st.button("Run Ranking System"):
    with st.spinner("Running ranking pipeline... This may take a few seconds."):
        # Run the rank.py script via subprocess
        try:
            result = subprocess.run(
                ["python", "rank.py", "--candidates", "data/jobs.json", "--out", "submission.csv"],
                capture_output=True,
                text=True,
                check=True
            )
            
            st.success("Ranking completed successfully!")
            
            # Show the output of the script if needed
            with st.expander("View Execution Logs"):
                st.code(result.stdout)
                
            # Load and display the CSV
            if os.path.exists("submission.csv"):
                df = pd.read_csv("submission.csv")
                st.subheader("Ranked Candidates (submission.csv)")
                st.dataframe(df)
                
                # Provide a download button
                with open("submission.csv", "rb") as file:
                    st.download_button(
                        label="Download submission.csv",
                        data=file,
                        file_name="submission.csv",
                        mime="text/csv"
                    )
            else:
                st.error("submission.csv was not generated.")
                
        except subprocess.CalledProcessError as e:
            st.error("An error occurred while running the ranking system.")
            st.code(e.stderr)
