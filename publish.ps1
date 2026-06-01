#!/usr/bin/env pwsh
# Quick publish script for ua-stress-engine
# Usage: .\publish.ps1 [test|prod]

param(
    [Parameter(Position=0)]
    [ValidateSet('test', 'prod')]
    [string]$Target = 'test'
)

$ErrorActionPreference = "Stop"

Write-Host "🚀 Publishing ua-stress-engine to $Target" -ForegroundColor Cyan

# Check prerequisites
Write-Host "`n📋 Checking prerequisites..." -ForegroundColor Yellow

# Check Rust
if (-not (Get-Command rustc -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Rust not found. Install from https://rustup.rs/" -ForegroundColor Red
    exit 1
}
Write-Host "✅ Rust: $(rustc --version)" -ForegroundColor Green

# Check maturin
if (-not (Get-Command maturin -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Maturin not found. Installing..." -ForegroundColor Yellow
    pip install maturin
}
Write-Host "✅ Maturin: $(maturin --version)" -ForegroundColor Green

# Check data file
$dataFile = "data\processed\ua_stress.bin.bz2"
if (-not (Test-Path $dataFile)) {
    Write-Host "❌ Data file missing: $dataFile" -ForegroundColor Red
    Write-Host "   Run data preparation scripts first." -ForegroundColor Yellow
    exit 1
}
Write-Host "✅ Data file: $dataFile" -ForegroundColor Green

# Read version
$version = (Select-String -Path "pyproject.toml" -Pattern 'version = "(.+)"').Matches[0].Groups[1].Value
Write-Host "✅ Version: $version" -ForegroundColor Green

# Confirm
Write-Host "`n⚠️  Publishing version $version to $Target" -ForegroundColor Yellow
$confirmation = Read-Host "Continue? (y/n)"
if ($confirmation -ne 'y') {
    Write-Host "Cancelled." -ForegroundColor Red
    exit 0
}

# Clean previous builds
Write-Host "`n🧹 Cleaning previous builds..." -ForegroundColor Yellow
if (Test-Path "target\wheels") {
    Remove-Item -Recurse -Force "target\wheels"
}
if (Test-Path "dist") {
    Remove-Item -Recurse -Force "dist"
}

# Build
Write-Host "`n🔨 Building wheels..." -ForegroundColor Yellow
maturin build --release
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Build failed" -ForegroundColor Red
    exit 1
}

# Test wheel locally
Write-Host "`n🧪 Testing wheel locally..." -ForegroundColor Yellow
$wheel = Get-ChildItem "target\wheels\*.whl" | Select-Object -First 1
if (-not $wheel) {
    Write-Host "❌ No wheel found" -ForegroundColor Red
    exit 1
}

Write-Host "Installing: $($wheel.Name)" -ForegroundColor Cyan
pip install --force-reinstall $wheel.FullName

Write-Host "Testing basic functionality..." -ForegroundColor Cyan
$testResult = python -c "import ukrainian_stress; print(ukrainian_stress.mark('мама'))"
if ($testResult -ne "ма́ма") {
    Write-Host "❌ Test failed. Expected: ма́ма, Got: $testResult" -ForegroundColor Red
    exit 1
}
Write-Host "✅ Test passed: $testResult" -ForegroundColor Green

# Publish
Write-Host "`n📦 Publishing to PyPI..." -ForegroundColor Yellow

if ($Target -eq 'test') {
    Write-Host "Publishing to TestPyPI..." -ForegroundColor Cyan
    maturin publish --repository testpypi --username __token__
} else {
    Write-Host "Publishing to PyPI..." -ForegroundColor Cyan
    maturin publish --username __token__
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Publish failed" -ForegroundColor Red
    exit 1
}

Write-Host "`n✅ Successfully published ua-stress-engine v$version to $Target!" -ForegroundColor Green

# Next steps
Write-Host "`n📝 Next steps:" -ForegroundColor Yellow
Write-Host "  1. Create git tag: git tag v$version && git push --tags"
Write-Host "  2. Create GitHub release"
if ($Target -eq 'test') {
    Write-Host "  3. Test installation: pip install -i https://test.pypi.org/simple/ ua-stress-engine"
} else {
    Write-Host "  3. Test installation: pip install ua-stress-engine"
}
