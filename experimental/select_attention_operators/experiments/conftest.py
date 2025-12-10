# experiments/conftest.py
import sys
import os

# Add each subdirectory of experiments to sys.path
base = os.path.dirname(__file__)
for name in os.listdir(base):
    subdir = os.path.join(base, name)
    if os.path.isdir(subdir):
        sys.path.insert(0, subdir)
