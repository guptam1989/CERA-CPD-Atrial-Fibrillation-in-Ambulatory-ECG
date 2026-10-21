@echo off
setlocal
pushd "%~dp0"
if errorlevel 1 exit /b 1
if not exist "build" mkdir "build"
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
if errorlevel 1 goto failed
bibtex build/main
if errorlevel 1 goto failed
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
if errorlevel 1 goto failed
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
if errorlevel 1 goto failed
move /Y "build\main.pdf" "CERA-CPD_manuscript_updated.pdf" >nul
if errorlevel 1 goto failed
echo Updated PDF: CERA-CPD_manuscript_updated.pdf
popd
exit /b 0
:failed
echo Build failed. See build\main.log for details.
popd
exit /b 1
