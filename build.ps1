# Configure + build gpu-qsim (Release), then optionally run the tests.
#
#   .\build.ps1            configure, build everything
#   .\build.ps1 -Test      ...and run the Catch2 suite (and pytest if the Python module was built)
#   .\build.ps1 -Clean     wipe build/ first
#
# Uses the CMake bundled with VS Build Tools on purpose: the WinLibs cmake on PATH has a curl with no
# CA certificates, so FetchContent downloads fail with "SSL peer certificate ... was not OK".
# Also reloads CUDA_PATH* from the machine environment, so shells opened before the CUDA install work.
# The Python module (python/qsim/_qsim*.pyd) is built when .venv exists and has nanobind installed.

param([switch]$Test, [switch]$Clean)
$ErrorActionPreference = 'Stop'

$cmake = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
$ctest = Join-Path (Split-Path $cmake) 'ctest.exe'
if (-not (Test-Path $cmake)) { throw "VS Build Tools CMake not found at $cmake" }

foreach ($k in 'CUDA_PATH', 'CUDA_PATH_V13_1') {
  $v = [Environment]::GetEnvironmentVariable($k, 'Machine')
  if ($v) { Set-Item "env:$k" $v }
}

$root = $PSScriptRoot
$build = Join-Path $root 'build'
if ($Clean -and (Test-Path $build)) { Remove-Item -Recurse -Force $build }

$cfg = @('-S', $root, '-B', $build, '-G', 'Visual Studio 17 2022', '-A', 'x64', '-T', 'cuda=13.1')
$py = Join-Path $root '.venv\Scripts\python.exe'
$withPython = $false
if (Test-Path $py) {
  & $py -c "import nanobind" 2>$null
  $withPython = ($LASTEXITCODE -eq 0)
}
if ($withPython) { $cfg += @('-DQSIM_BUILD_PYTHON=ON', "-DPython_EXECUTABLE=$py") }

& $cmake @cfg | Out-Null
if ($LASTEXITCODE) { & $cmake @cfg; throw 'configure failed' }
& $cmake --build $build --config Release --parallel
if ($LASTEXITCODE) { throw 'build failed' }

if ($Test) {
  & $ctest --test-dir $build -C Release --output-on-failure
  if ($withPython) { & $py -m pytest (Join-Path $root 'tests\test_python.py') -q }
}
