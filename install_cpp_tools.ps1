# PowerShell script để cài đặt MinGW-w64 (G++) trên Windows
Write-Host "Checking for G++ installation..."

# Kiểm tra xem g++ đã được cài đặt chưa
$gppExists = $null -ne (Get-Command g++ -ErrorAction SilentlyContinue)

if (-not $gppExists) {
    Write-Host "G++ not found. Installing MinGW-w64..."
    
    # Tạo thư mục temp nếu chưa tồn tại
    $tempDir = "C:\temp"
    if (-not (Test-Path $tempDir)) {
        New-Item -ItemType Directory -Path $tempDir | Out-Null
    }
    
    # Download MinGW-w64 installer
    $installerUrl = "https://github.com/brechtsanders/winlibs_mingw/releases/download/13.2.0-16.0.6-11.0.0-ucrt-r1/winlibs-x86_64-posix-seh-gcc-13.2.0-mingw-w64ucrt-11.0.0-r1.zip"
    $installerPath = Join-Path $tempDir "mingw64.zip"
    
    Write-Host "Downloading MinGW-w64..."
    Invoke-WebRequest -Uri $installerUrl -OutFile $installerPath
    
    # Extract MinGW-w64
    Write-Host "Extracting MinGW-w64..."
    Expand-Archive -Path $installerPath -DestinationPath "C:\mingw64" -Force
    
    # Thêm vào Path
    Write-Host "Adding MinGW-w64 to PATH..."
    $mingwPath = "C:\mingw64\mingw64\bin"
    $currentPath = [Environment]::GetEnvironmentVariable("Path", "User")
    if ($currentPath -notlike "*$mingwPath*") {
        [Environment]::SetEnvironmentVariable("Path", "$currentPath;$mingwPath", "User")
    }
    
    # Cleanup
    Remove-Item $installerPath
    
    Write-Host "MinGW-w64 installation completed. Please restart your terminal to use g++."
} else {
    Write-Host "G++ is already installed."
}

# Kiểm tra phiên bản
Write-Host "Checking G++ version..."
g++ --version
