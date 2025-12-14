#!/usr/bin/env python3
"""
Simple script to train and evaluate the MNIST model.
Run: python run_mnist.py
"""

import subprocess
import sys

def main():
    print("🚀 Training MNIST model...")
    
    # Convert notebook to script and run
    try:
        subprocess.run([
            sys.executable, "-m", "jupyter", "nbconvert",
            "--to", "script", "main.ipynb",
            "--output", "temp_script"
        ], check=True)
        
        # Run the converted script
        subprocess.run([sys.executable, "temp_script.py"], check=True)
        
        print("✅ Training complete! Check main.ipynb for results.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please run the notebook manually: jupyter notebook main.ipynb")

if __name__ == "__main__":
    main()