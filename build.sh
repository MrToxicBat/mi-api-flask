#!/usr/bin/env bash
pip install --upgrade pip setuptools wheel
pip uninstall -y google-generativeai google-ai-generativelanguage || true
pip install google-generativeai==0.3.2
