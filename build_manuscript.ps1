param(
  [string]$Python = 'python',
  [string]$Tectonic = 'tectonic',
  [string]$TectonicCache = '',
  [switch]$Offline,
  [switch]$CompileOnly,
  [ValidateSet('Auto','Tectonic','PdfLaTeX','XeLaTeX')]
  [string]$Engine = 'Auto'
)
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
Push-Location $projectRoot
try {
  if (-not $CompileOnly) {
  & $Python scripts/shape_spectral_examples.py
  if ($LASTEXITCODE -ne 0) { throw 'Numerical study failed.' }
  & $Python scripts/verify_certificates.py
  if ($LASTEXITCODE -ne 0) { throw 'Saved certificate verification failed.' }
  & $Python scripts/verify_sharp.py
  if ($LASTEXITCODE -ne 0) { throw 'Boundary/gap/structured-remainder verification failed.' }
  & $Python scripts/verify_coupled.py
  if ($LASTEXITCODE -ne 0) { throw 'Water/global-maximum verification failed.' }
  & $Python scripts/verify_extensions.py
  if ($LASTEXITCODE -ne 0) { throw 'Spatial-water/broad-kernel/finite-rank verification failed.' }
  & $Python scripts/verify_coupled_optimum.py
  if ($LASTEXITCODE -ne 0) { throw 'Coupled all-integer optimum/ratio-sector verification failed.' }
  & $Python scripts/verify_second_order.py
  if ($LASTEXITCODE -ne 0) { throw 'Weak-measure/cubic-budget verification failed.' }
  & $Python scripts/verify_transitions.py
  if ($LASTEXITCODE -ne 0) { throw 'Local all-integer transition verification failed.' }
  & $Python scripts/verify_signed_second.py
  if ($LASTEXITCODE -ne 0) { throw 'Signed second-coefficient/energy verification failed.' }
  & $Python scripts/verify_cubic_anchor.py
  if ($LASTEXITCODE -ne 0) { throw 'Anchor-dependent cubic-budget verification failed.' }
  & $Python scripts/verify_predictor.py
  if ($LASTEXITCODE -ne 0) { throw 'Envelope-predictor pointwise-error verification failed.' }
  & $Python scripts/verify_hilbert_remainder.py
  if ($LASTEXITCODE -ne 0) { throw 'Uniform Hilbert-space cubic-remainder verification failed.' }
  }
  if ($TectonicCache) { $env:TECTONIC_CACHE_DIR = $TectonicCache }
  New-Item -ItemType Directory -Force output/pdf | Out-Null
  # The main article and the Supplementary material cross-reference each other
  # through xr-hyper, so both are compiled in the project directory (each reads
  # the other's .aux) in the order main, supplement, main, supplement, main.
  # Tectonic is not used.
  if ($Engine -eq 'Auto' -or $Engine -eq 'Tectonic') {
    if (Get-Command pdflatex -ErrorAction SilentlyContinue) { $Engine = 'PdfLaTeX' }
    elseif (Get-Command xelatex -ErrorAction SilentlyContinue) { $Engine = 'XeLaTeX' }
    else { throw 'No TeX engine found. Install pdfLaTeX or XeLaTeX.' }
  }
  $texProgram = if ($Engine -eq 'PdfLaTeX') { 'pdflatex' } else { 'xelatex' }
  $texExecutable = (Get-Command $texProgram -ErrorAction Stop).Source
  $baseArguments = @('-interaction=nonstopmode','-halt-on-error','-file-line-error')
  if ($texExecutable -match 'MiKTeX') { $baseArguments = @('--disable-installer') + $baseArguments }
  foreach ($doc in @('paper_journal','paper_OR1','paper_journal','paper_OR1','paper_journal')) {
    # Some TeX installations print harmless warnings on stderr; judge success by
    # the exit code only.
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    & $texProgram @($baseArguments + @($doc + '.tex')) 2>&1 | Out-Null
    $code = $LASTEXITCODE
    $ErrorActionPreference = $previous
    if ($code -ne 0) { throw ($texProgram + ' compilation failed for ' + $doc) }
  }
  Copy-Item paper_journal.pdf, paper_OR1.pdf output/pdf -Force
} finally {
  Pop-Location
}
