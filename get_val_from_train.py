#!/usr/bin/env python3

import json
import glob

"""
This script basically gets all files used for training by
looking at the validation files list.
"""

with open("validation_files.json") as f: files = set(json.load(f))
with open("training_files.json", "w") as f: json.dump(list(set(glob.glob("dataset/MICCAI_BraTS_2019_Data_Training/*GG/*")) - files), f, indent=2)