#!/usr/bin/env bash
pip install --upgrade pip setuptools wheel
pip uninstall -y google-generativeai google-ai-generativelanguage || true
pip install -r requirements.txt
pip install --upgrade google-generativeai==0.3.2
