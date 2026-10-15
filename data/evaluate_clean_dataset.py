#!/usr/bin/env python3
"""Run the self-contained release benchmark."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modeling.evaluation import main

if __name__ == "__main__":
    main()
