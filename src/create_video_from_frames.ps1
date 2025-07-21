# create_video_from_frames.ps1
#
# A script to find all 'train' and 'val' frames for a specific video ID
# in the DeepFish dataset and combine them into a single video file,
# saving the output directly into the video's ID folder.
#
# USAGE:
# 1. Open PowerShell in the project's root directory.
# 2. Run the script with the VideoID you want to process:
#    .\create_video_from_frames.ps1 -VideoID 7117

param (
    [Parameter(Mandatory=$true)]
    [string]$VideoID,

    [string]$DatasetPath = "..\Datasets\Deepfish"
)

# --- Script Logic ---

Write-Host "🚀 Starting video creation for Video ID: $VideoID..." -ForegroundColor Green

# 1. Define paths
$FullDatasetPath = Join-Path -Path $PSScriptRoot -ChildPath $DatasetPath
$VideoIDPath = Join-Path -Path $FullDatasetPath -ChildPath $VideoID
# Temporary files will still go into an outputs folder for easy cleanup
$TempOutputPath = Join-Path -Path $PSScriptRoot -ChildPath "outputs\videos"
$TempFramePath = Join-Path -Path $TempOutputPath -ChildPath "temp_frames_$VideoID"


# Check if paths are valid
if (-not (Test-Path $VideoIDPath)) {
    Write-Host "❌ Error: Video ID folder '$VideoID' not found at '$VideoIDPath'." -ForegroundColor Red
    exit
}

# 2. Create temporary directory
New-Item -ItemType Directory -Force -Path $TempFramePath | Out-Null

# 3. Find and copy all frames
Write-Host "🔍 Finding and copying frames..."
$frames = Get-ChildItem -Path $VideoIDPath -Recurse -Filter "$($VideoID)_*.jpg"
if ($frames.Count -eq 0) {
    Write-Host "❌ Error: No .jpg frames starting with '$($VideoID)_' found in '$VideoIDPath'." -ForegroundColor Red
    Remove-Item -Path $TempFramePath -Recurse
    exit
}

$frames | Sort-Object Name | ForEach-Object {
    Copy-Item -Path $_.FullName -Destination $TempFramePath
}
Write-Host "✅ Copied $($frames.Count) frames to a temporary directory."

# 4. Create the file list for FFmpeg
Write-Host "📝 Creating file list for FFmpeg..."
$fileListPath = Join-Path -Path $TempFramePath -ChildPath "mylist.txt"
$filesInTemp = Get-ChildItem -Path $TempFramePath -Filter "*.jpg" | Sort-Object Name
$fileListContent = $filesInTemp | ForEach-Object { "file '$($_.Name)'" }

if ($fileListContent.Count -eq 0) {
    Write-Host "❌ Error: The list of files to process is empty." -ForegroundColor Red
    Remove-Item -Path $TempFramePath -Recurse
    exit
}

$encoding = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllLines($fileListPath, $fileListContent, $encoding)

# 5. Run FFmpeg to create the video
$OutputVideoPath = Join-Path -Path $VideoIDPath -ChildPath "$($VideoID).mp4"
Write-Host "🎬 Running FFmpeg to create video at '$OutputVideoPath'..."

ffmpeg -r 30 -f concat -safe 0 -i $fileListPath -c:v libx264 -pix_fmt yuv420p $OutputVideoPath

if ($?) {
    Write-Host "✅ Video created successfully!" -ForegroundColor Green
} else {
    Write-Host "❌ Error: FFmpeg failed to create the video." -ForegroundColor Red
}

# 6. Cleanup
Write-Host "🧹 Cleaning up temporary files..."
Remove-Item -Path $TempFramePath -Recurse

Write-Host "🎉 Process complete."