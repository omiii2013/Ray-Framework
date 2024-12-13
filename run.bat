@echo off
echo Installing Python...

:: Check if Python is installed
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo Python is not installed. Installing Python...

    :: Download Python 3.10.11 installer
    powershell -Command "Invoke-WebRequest -Uri https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe -OutFile python-installer.exe"

    :: Install Python 3.10.11 silently
    python-installer.exe /quiet InstallAllUsers=1 PrependPath=1

    :: Check again if Python is installed successfully
    where python >nul 2>nul
    if %errorlevel% neq 0 (
        echo Python installation failed. Exiting...
        pause
        exit /b
    )
    echo Python 3.10.11 installed successfully!
    del python-installer.exe
) else (
    echo Python is already installed.
)

echo Installing Ray...
pip install ray
echo Ray Full Installation Complete.

echo Verifying Ray Installation...
python -c "import ray; print('Ray version:', ray.__version__)"
if %errorlevel%==0 (
    echo Ray is successfully installed and running!
) else (
    echo There was an issue initializing Ray. Please check the logs above.
    pause
    exit /b

)
    

:: Run the first Python script
python XGBoost_Algorithm_Ray.py
if %errorlevel% neq 0 (
    echo Error occurred while running XGBoost_Algorithm_Ray.py
    pause
    exit /b
)

:: Run the second Python script
python deep_learning_model_ray_implementation.py
if %errorlevel% neq 0 (
    echo Error occurred while running deep_learning_model_ray_implementation.py
    pause
    exit /b
)

echo All Python files executed successfully!
pause
