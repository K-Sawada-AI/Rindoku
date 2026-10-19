@echo off
cd /d "%~dp0"
start "" http://localhost:8000/week2.html
node server.js
