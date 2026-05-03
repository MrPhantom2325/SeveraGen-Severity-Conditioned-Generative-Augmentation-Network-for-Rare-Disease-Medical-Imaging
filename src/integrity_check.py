# src/integrity_check.py
# Scans all images and removes corrupted ones before training
# Run this once — takes about 30 seconds

import os
from PIL import Image
from tqdm import tqdm

DATA_DIR   = 'data/processed'
SEVERITIES = ['mild', 'moderate', 'severe']

corrupt   = []
too_small = []
ok        = 0

print("Scanning all images for corruption and size issues...\n")

for sev in SEVERITIES:
    folder = os.path.join(DATA_DIR, sev)
    files  = [f for f in os.listdir(folder)
              if f.lower().endswith(('.jpg','.jpeg','.png'))]

    for fname in tqdm(files, desc=sev):
        fpath = os.path.join(folder, fname)
        try:
            img = Image.open(fpath)
            img.verify()            # catches truncated / corrupted files
            img = Image.open(fpath) # re-open after verify (verify closes it)
            w, h = img.size
            if w < 32 or h < 32:
                too_small.append(fpath)
        except Exception as e:
            corrupt.append((fpath, str(e)))

print(f"\n{'='*45}")
print(f"Scan complete")
print(f"  OK          : {sum(len(os.listdir(os.path.join(DATA_DIR,s))) for s in SEVERITIES) - len(corrupt) - len(too_small)}")
print(f"  Corrupted   : {len(corrupt)}")
print(f"  Too small   : {len(too_small)}")

if corrupt:
    print("\nCorrupted files (will be removed):")
    for path, err in corrupt:
        print(f"  {path} — {err}")
        os.remove(path)
    print("Removed all corrupted files.")

if too_small:
    print("\nToo-small files (will be removed):")
    for path in too_small:
        print(f"  {path}")
        os.remove(path)
    print("Removed all undersized files.")

if not corrupt and not too_small:
    print("\nAll images are clean. Ready for training.")