import os
import zipfile
from datetime import datetime

root = r"c:\Users\ashwinanilkumar.p\Ashwin\Incident Trend Analysis"
parent = os.path.dirname(root)
zip_name = os.path.join(parent, f"Incident_Trend_Analysis_{datetime.now():%Y%m%d}.zip")
# Exclude common large or environment folders to keep archive small
exclude_dirs = {"venv", ".git", ".vs", "__pycache__", "node_modules", "workspaceStorage"}

with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as z:
    for folder, dirs, files in os.walk(root):
        # modify dirs in-place to skip excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for f in files:
            fp = os.path.join(folder, f)
            # skip files inside explicitly excluded path components
            if any(part in exclude_dirs for part in fp.split(os.sep)):
                continue
            arcname = os.path.relpath(fp, parent)
            z.write(fp, arcname)

print(zip_name)
