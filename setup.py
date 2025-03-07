import os
import subprocess

def build_executable():
    """Build the executable using PyInstaller"""
    print("Building executable with PyInstaller...")
    
    # Basic PyInstaller command
    cmd = [
        "pyinstaller",
        "--onefile",  # Create a single executable file
        "--windowed",  # Don't show console window (for GUI apps)
        "--name", "ImageSimilarityContest",
        "image_similarity_app.py"  # Your main script
    ]
    
    # Additional data files if needed
    # cmd.extend(["--add-data", "path/to/file:destination/in/app"])
    
    # Run the PyInstaller command
    subprocess.run(cmd)
    print("Build completed. Check the 'dist' folder for your executable.")

if __name__ == "__main__":
    build_executable()