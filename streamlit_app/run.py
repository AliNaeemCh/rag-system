# run_streamlit.py

streamlit_cmd = [
    "streamlit",
    "run",
    "streamlit_app/main.py"
]

if __name__ == "__main__":
    import subprocess

    subprocess.run(streamlit_cmd)