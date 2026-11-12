@echo off
rem Lance TurboWatch : API FastAPI puis ouverture de l'interface dans Edge.
rem Double-cliquer sur ce fichier. Fermer la fenetre pour arreter l'API.
cd /d "%~dp0"
title TurboWatch - API (fermer cette fenetre pour arreter)
if not exist "venv\Scripts\activate.bat" (
  echo Environnement virtuel introuvable : creez-le avec  py -3.12 -m venv venv
  pause
  exit /b 1
)
call "venv\Scripts\activate.bat"
echo Demarrage de l'API TurboWatch... Edge s'ouvrira automatiquement des qu'elle sera prete.
start "" /b powershell -NoProfile -WindowStyle Hidden -Command "do { Start-Sleep -Milliseconds 700; try { $ok = (Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 http://127.0.0.1:8000/health).StatusCode -eq 200 } catch { $ok = $false } } until ($ok); Start-Process msedge http://127.0.0.1:8000/ui"
uvicorn app:app --host 127.0.0.1 --port 8000
