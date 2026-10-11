@echo off
if "%CLIMB_API_URL%"=="" (
    echo Set CLIMB_API_URL to the deployed Modal API URL first.
    exit /b 1
)

set /p GRADE="Enter grade (e.g., 7A): "
set /p START_X="Enter start column number (0-10): "
set /p START_Y="Enter start row (1-17): "
set /p END_X="Enter end column number (0-10): "
set /p END_Y="Enter end row (1-17): "

curl -X POST "%CLIMB_API_URL%/api/v1/routes/generate" ^
    -H "Content-Type: application/json" ^
    -d "{\"grade\":\"%GRADE%\",\"start_hold\":[%START_X%,%START_Y%],\"end_hold\":[%END_X%,%END_Y%],\"candidate_count\":3}"
