"""
Main entry point for the Disaster Risk Prediction application.
Run this script to start the Streamlit app.
"""
import os
import subprocess
import sys

def main():
    """Run the Streamlit application"""
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set the working directory to the script directory
    os.chdir(script_dir)
    
    # Add the current directory to Python path
    sys.path.insert(0, script_dir)
    
    # Print current directory and Python path for debugging
    print(f"Current directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")
    
    # Run the Streamlit app
    subprocess.run(["streamlit", "run", "streamlit_app/app.py"])

if __name__ == "__main__":
    main()
