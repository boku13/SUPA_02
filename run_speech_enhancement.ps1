# PowerShell script to run the full speech separation and speaker identification pipeline
# Each step will output to its own log file

# Create timestamp for unique log file names
$TIMESTAMP = Get-Date -Format "yyyyMMdd_HHmmss"

# Create logs directory inside speech_enhancement
$LOGS_DIR = "src\speech_enhancement\logs"
New-Item -ItemType Directory -Force -Path $LOGS_DIR | Out-Null

# Define log file paths
$MIXTURES_TRAIN_LOG = "$LOGS_DIR\mixtures_train_$TIMESTAMP.log"
$MIXTURES_TEST_LOG = "$LOGS_DIR\mixtures_test_$TIMESTAMP.log"
$SEPARATION_LOG = "$LOGS_DIR\separation_$TIMESTAMP.log"
$EVALUATION_LOG = "$LOGS_DIR\evaluation_$TIMESTAMP.log"
$SUMMARY_LOG = "$LOGS_DIR\summary_$TIMESTAMP.log"

# Define directories and create them if they don't exist
New-Item -ItemType Directory -Force -Path "data\vox2\mixtures_train" | Out-Null
New-Item -ItemType Directory -Force -Path "data\vox2\mixtures_test" | Out-Null
New-Item -ItemType Directory -Force -Path "data\vox2\separated_test" | Out-Null
New-Item -ItemType Directory -Force -Path "results\speaker_separation" | Out-Null

# Define the path to VoxCeleb2 dataset
$VOX2_METADATA = "data\vox2\vox2_metadata.csv"
$VOX2_AUDIO_ROOT = "data\vox2\aac"

# Path to virtual environment activation script
$ACTIVATE_ENV = "speech\Scripts\activate"

# Ensure activation script exists
if (-not (Test-Path $ACTIVATE_ENV)) {
    Write-Host "Error: Virtual environment activation script not found at: $ACTIVATE_ENV" -ForegroundColor Red
    Write-Host "Please ensure the 'speech' virtual environment is correctly set up." -ForegroundColor Red
    exit 1
}

# Start with a summary header
"===== SPEECH ENHANCEMENT AND SPEAKER SEPARATION PIPELINE =====" | Out-File -FilePath $SUMMARY_LOG
"Started at: $(Get-Date)" | Out-File -FilePath $SUMMARY_LOG -Append
"" | Out-File -FilePath $SUMMARY_LOG -Append

# Function to run a process and update the summary log
function Run-Step {
    param (
        [string]$StepName,
        [string]$Command,
        [string]$LogFile
    )
    
    # Log step start
    "[$(Get-Date -Format 'HH:mm:ss')] Starting: $StepName" | Tee-Object -FilePath $SUMMARY_LOG -Append
    
    # Run the command with activated environment and log output
    # Use cmd /c to run the command with the virtual environment activated
    $fullCommand = "cmd /c `"$ACTIVATE_ENV && $Command`""
    Invoke-Expression "$fullCommand *> $LogFile"
    
    # Log step completion
    "[$(Get-Date -Format 'HH:mm:ss')] Completed: $StepName" | Tee-Object -FilePath $SUMMARY_LOG -Append
    "Log file: $LogFile" | Out-File -FilePath $SUMMARY_LOG -Append
    "" | Out-File -FilePath $SUMMARY_LOG -Append
}

# Step 1a: Create multi-speaker mixtures for training (first 50 speakers)
$command1a = "python src\speech_enhancement\create_mixtures.py --metadata_file '$VOX2_METADATA' --audio_root '$VOX2_AUDIO_ROOT' --output_dir 'data\vox2\mixtures_train' --speaker_start 0 --num_speakers 50 --num_mixtures 500 --snr_min -5 --snr_max 5"
Run-Step -StepName "Creating training mixtures (first 50 speakers)" -Command $command1a -LogFile $MIXTURES_TRAIN_LOG

# Step 1b: Create multi-speaker mixtures for testing (next 50 speakers)
$command1b = "python src\speech_enhancement\create_mixtures.py --metadata_file '$VOX2_METADATA' --audio_root '$VOX2_AUDIO_ROOT' --output_dir 'data\vox2\mixtures_test' --speaker_start 50 --num_speakers 50 --num_mixtures 200 --snr_min -5 --snr_max 5"
Run-Step -StepName "Creating testing mixtures (next 50 speakers)" -Command $command1b -LogFile $MIXTURES_TEST_LOG

# Step 2: Perform speaker separation using SepFormer
$command2 = "python src\speech_enhancement\sepformer_direct.py --input_dir 'data\vox2\mixtures_test' --output_dir 'data\vox2\separated_test' --metadata_file 'data\vox2\mixtures_test\metadata.csv'"
Run-Step -StepName "Performing speaker separation using SepFormer" -Command $command2 -LogFile $SEPARATION_LOG

# Step 3: Evaluate separation results and run speaker identification
$command3 = "python src\speech_enhancement\run_separation.py --test_metadata 'data\vox2\mixtures_test\metadata.csv' --separated_dir 'data\vox2\separated_test' --output_dir 'results\speaker_separation' --model_name 'wavlm_base_plus'"
Run-Step -StepName "Evaluating separation and identifying speakers" -Command $command3 -LogFile $EVALUATION_LOG

# Extract and include summary metrics in the summary log
if (Test-Path "results\speaker_separation\summary.txt") {
    "===== PERFORMANCE METRICS SUMMARY =====" | Out-File -FilePath $SUMMARY_LOG -Append
    Get-Content "results\speaker_separation\summary.txt" | Out-File -FilePath $SUMMARY_LOG -Append
    "" | Out-File -FilePath $SUMMARY_LOG -Append
}

"===== PIPELINE COMPLETED =====" | Out-File -FilePath $SUMMARY_LOG -Append
"Completed at: $(Get-Date)" | Out-File -FilePath $SUMMARY_LOG -Append
"Results are available in: results\speaker_separation\" | Out-File -FilePath $SUMMARY_LOG -Append
"All logs are available in: $LOGS_DIR" | Out-File -FilePath $SUMMARY_LOG -Append

Write-Host "Speech Enhancement Pipeline completed!"
Write-Host "Summary log saved to: $SUMMARY_LOG"
Write-Host "All detailed logs are in: $LOGS_DIR" 