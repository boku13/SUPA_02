@echo off
setlocal enabledelayedexpansion

:: Script to run the full speech separation and speaker identification pipeline in the background
:: Each step will output to its own log file

:: Create timestamp for unique log file names
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /format:list') do set datetime=%%I
set TIMESTAMP=%datetime:~0,8%_%datetime:~8,6%

:: Create logs directory inside speech_enhancement
set LOGS_DIR=src\speech_enhancement\logs
if not exist %LOGS_DIR% mkdir %LOGS_DIR%

:: Define log file paths
set MIXTURES_TRAIN_LOG=%LOGS_DIR%\mixtures_train_%TIMESTAMP%.log
set MIXTURES_TEST_LOG=%LOGS_DIR%\mixtures_test_%TIMESTAMP%.log
set SEPARATION_LOG=%LOGS_DIR%\separation_%TIMESTAMP%.log
set EVALUATION_LOG=%LOGS_DIR%\evaluation_%TIMESTAMP%.log
set SUMMARY_LOG=%LOGS_DIR%\summary_%TIMESTAMP%.log

:: Define directories and create them if they don't exist
if not exist data\vox2\mixtures_train mkdir data\vox2\mixtures_train
if not exist data\vox2\mixtures_test mkdir data\vox2\mixtures_test
if not exist data\vox2\separated_test mkdir data\vox2\separated_test
if not exist results\speaker_separation mkdir results\speaker_separation

:: Define the path to VoxCeleb2 dataset
set VOX2_METADATA=data\vox2\vox2_metadata.csv
set VOX2_AUDIO_ROOT=data\vox2\aac

:: Path to virtual environment activation script
set ACTIVATE_ENV=speech\Scripts\activate

:: Check if activation script exists
if not exist %ACTIVATE_ENV% (
    echo Error: Virtual environment activation script not found at: %ACTIVATE_ENV%
    echo Please ensure the 'speech' virtual environment is correctly set up.
    exit /b 1
)

:: Create a background task that will run the entire pipeline
echo Creating background task for speech enhancement pipeline...

:: Write to the summary log
echo ===== SPEECH ENHANCEMENT AND SPEAKER SEPARATION PIPELINE ===== > %SUMMARY_LOG%
echo Started at: %date% %time% >> %SUMMARY_LOG%
echo. >> %SUMMARY_LOG%

:: Start the background process
start /B "Speech Enhancement Pipeline" cmd /c ^
(^
  echo [%time%] Starting Step 1a: Creating training mixtures ^(first 50 speakers^) >> %SUMMARY_LOG% ^
  && call %ACTIVATE_ENV% ^
  && python src\speech_enhancement\create_mixtures.py --metadata_file "%VOX2_METADATA%" --audio_root "%VOX2_AUDIO_ROOT%" --output_dir "data\vox2\mixtures_train" --speaker_start 0 --num_speakers 50 --num_mixtures 500 --snr_min -5 --snr_max 5 > %MIXTURES_TRAIN_LOG% 2>&1 ^
  && echo [%time%] Completed Step 1a: Training mixtures created >> %SUMMARY_LOG% ^
  && echo Log file: %MIXTURES_TRAIN_LOG% >> %SUMMARY_LOG% ^
  && echo. >> %SUMMARY_LOG% ^
  && echo [%time%] Starting Step 1b: Creating testing mixtures ^(next 50 speakers^) >> %SUMMARY_LOG% ^
  && python src\speech_enhancement\create_mixtures.py --metadata_file "%VOX2_METADATA%" --audio_root "%VOX2_AUDIO_ROOT%" --output_dir "data\vox2\mixtures_test" --speaker_start 50 --num_speakers 50 --num_mixtures 200 --snr_min -5 --snr_max 5 > %MIXTURES_TEST_LOG% 2>&1 ^
  && echo [%time%] Completed Step 1b: Testing mixtures created >> %SUMMARY_LOG% ^
  && echo Log file: %MIXTURES_TEST_LOG% >> %SUMMARY_LOG% ^
  && echo. >> %SUMMARY_LOG% ^
  && echo [%time%] Starting Step 2: Performing speaker separation using SepFormer >> %SUMMARY_LOG% ^
  && python src\speech_enhancement\sepformer_direct.py --input_dir "data\vox2\mixtures_test" --output_dir "data\vox2\separated_test" --metadata_file "data\vox2\mixtures_test\metadata.csv" > %SEPARATION_LOG% 2>&1 ^
  && echo [%time%] Completed Step 2: Speaker separation completed >> %SUMMARY_LOG% ^
  && echo Log file: %SEPARATION_LOG% >> %SUMMARY_LOG% ^
  && echo. >> %SUMMARY_LOG% ^
  && echo [%time%] Starting Step 3: Evaluating separation and identifying speakers >> %SUMMARY_LOG% ^
  && python src\speech_enhancement\run_separation.py --test_metadata "data\vox2\mixtures_test\metadata.csv" --separated_dir "data\vox2\separated_test" --output_dir "results\speaker_separation" --model_name "wavlm_base_plus" > %EVALUATION_LOG% 2>&1 ^
  && echo [%time%] Completed Step 3: Evaluation and speaker identification completed >> %SUMMARY_LOG% ^
  && echo Log file: %EVALUATION_LOG% >> %SUMMARY_LOG% ^
  && echo. >> %SUMMARY_LOG% ^
  && if exist "results\speaker_separation\summary.txt" ^( ^
     echo ===== PERFORMANCE METRICS SUMMARY ===== >> %SUMMARY_LOG% ^
     && type "results\speaker_separation\summary.txt" >> %SUMMARY_LOG% ^
     && echo. >> %SUMMARY_LOG% ^
  ^) ^
  && echo ===== PIPELINE COMPLETED ===== >> %SUMMARY_LOG% ^
  && echo Completed at: %date% %time% >> %SUMMARY_LOG% ^
  && echo Results are available in: results\speaker_separation\ >> %SUMMARY_LOG% ^
  && echo All logs are available in: %LOGS_DIR% >> %SUMMARY_LOG% ^
)

echo Speech Enhancement Pipeline started in the background
echo Summary log will be saved to: %SUMMARY_LOG%
echo All detailed logs will be in: %LOGS_DIR%
echo You can check progress with: type %SUMMARY_LOG%

endlocal 