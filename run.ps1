# Удобный запуск проекта на Windows (PowerShell).
# Использование:
#   .\run.ps1 seq            - последовательная версия
#   .\run.ps1 mpi 4          - MPI-версия на 4 процессах
#   .\run.ps1 bench          - бенчмарк ускорения
#   .\run.ps1 data           - только проверить/скачать датасет

param(
    [string]$cmd = "bench",
    [int]$procs = 4
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

switch ($cmd) {
    "data"  { python "$root\src\data_loader.py" }
    "seq"   { python "$root\src\recommender_seq.py" --users 3 }
    "mpi"   { mpiexec -n $procs python "$root\src\recommender_mpi.py" --users 3 }
    "bench" { python "$root\src\benchmark.py" }
    default { Write-Host "Неизвестная команда: $cmd. Доступно: data | seq | mpi | bench" }
}
