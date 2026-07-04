$DeployDir = "deploy_windows"
$WheelsDir = "$DeployDir\wheels"

If (!(Test-Path $DeployDir)) { New-Item -ItemType Directory -Force -Path $DeployDir }
If (!(Test-Path $WheelsDir)) { New-Item -ItemType Directory -Force -Path $WheelsDir }

# Download Python 3.11 installer
$PythonUrl = "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe"
$PythonExe = "$DeployDir\python-3.11.9-amd64.exe"
Write-Host "Downloading Python 3.11 installer for Windows..."
Invoke-WebRequest -Uri $PythonUrl -OutFile $PythonExe

# Download wheels
Write-Host "Downloading Python wheels for Windows..."
# Note: --only-binary=:all: ensures we only get wheels, no source dists that require compilation
pip download -d $WheelsDir --platform win_amd64 --python-version 3.11 --only-binary=:all: -r requirements.txt

# Copy application files
Write-Host "Copying application files..."
Copy-Item "app.py" -Destination $DeployDir -Force
Copy-Item "apply_filter_and_fluorescence.py" -Destination $DeployDir -Force
Copy-Item "fluorescence_k_data.py" -Destination $DeployDir -Force
Copy-Item "bundled_data.py" -Destination $DeployDir -Force
Copy-Item "ct_materials.py" -Destination $DeployDir -Force
Copy-Item "ct_optimizer.py" -Destination $DeployDir -Force
Copy-Item "ct_physics.py" -Destination $DeployDir -Force
Copy-Item "emission_optimizer.py" -Destination $DeployDir -Force
Copy-Item "test_fluorescence_physics.py" -Destination $DeployDir -Force
Copy-Item "*.csv" -Destination $DeployDir -Force
Copy-Item "pyne_edpl" -Destination $DeployDir -Recurse -Force
# Raw ENDF source files and metadata are not read at runtime
Remove-Item "$DeployDir\pyne_edpl\*.txt" -Force -ErrorAction SilentlyContinue
Remove-Item "$DeployDir\pyne_edpl\*_metadata.csv" -Force -ErrorAction SilentlyContinue
Copy-Item "install_and_run.bat" -Destination $DeployDir -Force
Copy-Item "requirements.txt" -Destination $DeployDir -Force

Write-Host "Windows packaging complete. Zip the '$DeployDir' directory."
