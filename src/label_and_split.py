import os, shutil, pandas as pd
import cv2, numpy as np
from tqdm import tqdm

df = pd.read_csv('/home/jovyan/work/rare-disease-cvae/data/raw/ISIC_2019_Training_GroundTruth.csv')
img_dir = '/home/jovyan/work/rare-disease-cvae/data/raw/ISIC_2019_Training_Input/ISIC_2019_Training_Input'
out_dir = '/home/jovyan/work/rare-disease-cvae/data/processed'

# Focus on rare classes only
rare = df[(df['DF'] == 1) | (df['VASC'] == 1)].reset_index(drop=True)
print(f"Total rare images: {len(rare)}")

def get_severity(img_path):
    """Proxy severity via colour variance — higher variance = more severe."""
    img = cv2.imread(img_path)
    if img is None:
        return 'moderate'
    variance = np.var(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1])
    if variance < 800:
        return 'mild'
    elif variance < 2000:
        return 'moderate'
    else:
        return 'severe'

for _, row in tqdm(rare.iterrows(), total=len(rare)):
    src = f"{img_dir}/{row['image']}.jpg"
    severity = get_severity(src)
    dst_dir = f"{out_dir}/{severity}"
    os.makedirs(dst_dir, exist_ok=True)
    shutil.copy(src, f"{dst_dir}/{row['image']}.jpg")

# Confirm distribution
for s in ['mild', 'moderate', 'severe']:
    count = len(os.listdir(f'{out_dir}/{s}'))
    print(f"{s}: {count} images")