# Script to run the Streamlit dashboard
# Requires: streamlit and matplotlib

Write-Host "Ensuring Streamlit and Matplotlib are installed..."
pip install streamlit matplotlib

Write-Host "Starting Dashboard..."
# Ensure the src directory is in the PYTHONPATH so module imports resolve correctly
$env:PYTHONPATH = "src"
streamlit run src\odx\ui\dashboard.py
