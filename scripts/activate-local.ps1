# Dot-source from PowerShell to use the repository-local toolchain on this workstation.
$promptshieldRoot = Split-Path -Parent $PSScriptRoot
$env:UV_CACHE_DIR = Join-Path $promptshieldRoot '.cache/uv'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $promptshieldRoot '.tools/python'
$env:TEMP = Join-Path $promptshieldRoot '.cache/tmp'
$env:TMP = $env:TEMP
$env:PYTHONIOENCODING = 'utf-8'
New-Item -ItemType Directory -Force $env:TEMP | Out-Null
$env:PATH = (Join-Path $promptshieldRoot '.tools/bin') + [IO.Path]::PathSeparator + $env:PATH
