# Run Nsight Compute elevated and save a report.
#
# GPU performance counters are left at the Windows default (admin-only) on purpose, so ncu has to
# run as admin. This wrapper does that through one UAC prompt, writes profiles/<Name>.ncu-rep plus
# the console output to profiles/<Name>.log, and prints the log. Reading a saved report doesn't need
# admin:  ncu --import profiles/<Name>.ncu-rep --page details
#
# Usage (everything after -Name goes to ncu; use absolute paths, the elevated shell starts elsewhere):
#   .\tools\ncu-admin.ps1 -Name probe24 --set full --kernel-name regex:"copy_f4|scale_inplace_f4" `
#       (Resolve-Path .\build\Release\bwprobe.exe) 24

param(
  [Parameter(Mandatory)][string]$Name,
  [Parameter(Mandatory, ValueFromRemainingArguments)][string[]]$Rest
)

$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$outDir = Join-Path $root 'profiles'
New-Item -ItemType Directory -Force $outDir | Out-Null

$ncu = Get-ChildItem 'C:\Program Files\NVIDIA Corporation\Nsight Compute*\ncu.bat' |
  Sort-Object FullName -Descending | Select-Object -First 1 -ExpandProperty FullName
if (-not $ncu) { throw 'Nsight Compute not found under C:\Program Files\NVIDIA Corporation' }

$rep = Join-Path $outDir $Name          # ncu appends .ncu-rep
$log = Join-Path $outDir "$Name.log"
if (Test-Path $log) { Remove-Item $log }

# Quote any argument with spaces or cmd metacharacters, then hand the whole line to an elevated cmd.
$quoted = ($Rest | ForEach-Object {
  if ($_ -match '[\s|&<>^"]') { '"' + ($_ -replace '"', '\"') + '"' } else { $_ }
}) -join ' '
$cmdLine = "/c `"`"$ncu`" -f -o `"$rep`" $quoted > `"$log`" 2>&1`""

$p = Start-Process cmd.exe -ArgumentList $cmdLine -Verb RunAs -Wait -PassThru -WindowStyle Hidden
if (Test-Path $log) { Get-Content $log }
"ncu exit code $($p.ExitCode); report: $rep.ncu-rep"
