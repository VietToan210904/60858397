@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.44
if errorlevel 1 exit /b %errorlevel%
"C:\Users\tonyh_yxuq8za\60858397_Evaluation\.venv-spar3d\Scripts\python.exe" -m pip install --ignore-installed --no-deps --no-build-isolation "C:\Users\tonyh_yxuq8za\60858397_Evaluation\vendor\SPAR3D\uv_unwrapper"
