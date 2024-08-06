import tkinter as tk
from tkinter import filedialog
import sys
import json


def select_folder():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    folder_path = filedialog.askdirectory()  # Open the dialog to select a folder
    if folder_path:
        print(json.dumps({"folder_path": folder_path}))
    root.destroy()

if __name__ == "__main__":
    select_folder()