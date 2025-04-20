#!/bin/bash
pip uninstall -y numpy
pip install numpy==1.24.4
exec gunicorn --bind=0.0.0.0:8000 app:app
