# Build script for native C++ extensions
# Requires: Visual Studio Build Tools (MSVC) and CMake.
# Make sure you have installed pybind11: pip install pybind11

Write-Host "Ensuring pybind11 is installed..."
pip install pybind11

$BuildDir = "cpp\build"

if (Test-Path $BuildDir) {
    Remove-Item -Recurse -Force $BuildDir
}
New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null

Set-Location $BuildDir

Write-Host "Configuring CMake (using MinGW Makefiles)..."
$PythonPath = (Get-Command python).Source
cmake -G "MinGW Makefiles" -DPython3_EXECUTABLE="$PythonPath" ..

cmake --build . --config Release

Set-Location ..\..
Write-Host "Build complete! bs_pricer_cpp should now be located in src\odx\pricers."
