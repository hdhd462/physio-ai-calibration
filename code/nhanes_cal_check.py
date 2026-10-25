#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR, PROJ_DIR, FIG_DIR, SUB_DIR, PATCH_DIR, NHANES_DIR, PROJ1_DIR, SHAOXING_DIR, PTBXL_DIR
import pandas as pd
import numpy as np
import os, glob

base = PROJ1_DIR
files = []
for root, dirs, fnames in os.walk(base):
    for f in fnames:
        if "predprob" in f.lower():
            files.append(os.path.join(root, f))

for fp in files:
    df = pd.read_csv(fp)
    print(f"\n=== {os.path.basename(fp)} ===")
    print(f"models: {df['model'].unique()}")
    print(f"dataset: {df['dataset'].unique()}")
    print(f"status: {df['status'].value_counts().to_dict()}")
    print(f"time range: {df['time'].min():.1f} - {df['time'].max():.1f}")
    print(f"prob(5yr) range: {df['5'].min():.4f} - {df['5'].max():.4f}, mean={df['5'].mean():.4f}")
    print(f"crank range: {df['crank'].min():.2f} - {df['crank'].max():.2f}")
