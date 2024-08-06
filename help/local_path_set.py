import streamlit as st
import subprocess
import json
import os

def main():
    st.title("Folder Selection and Management")

    if st.button("Select Folder"):
        result = subprocess.run(["python", "folder_selector.py"], capture_output=True, text=True)
        if result.returncode == 0:
            folder_data = json.loads(result.stdout)
            folder_path = folder_data.get("folder_path")
            if folder_path:
                st.success(f"Selected Folder: {folder_path}")
                required_folders = ['dataanalysis', 'database', 'process']
                missing_folders = [folder for folder in required_folders if not os.path.exists(os.path.join(folder_path, folder))]

                if not missing_folders:
                    st.write("All required folders already exist.")
                else:
                    for folder in missing_folders:
                        os.makedirs(os.path.join(folder_path, folder))
                    st.success(f"Created missing folders: {', '.join(missing_folders)}")

                # List files and folders in the selected folder as an example
                items = os.listdir(folder_path)
                st.write("Contents of the selected folder:")
                st.write(items)

                return folder_path

            else:
                st.error("No folder selected")
        else:
            st.error("Error selecting folder")


if __name__ == "__main__":
    main()