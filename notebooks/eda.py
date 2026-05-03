import os, pandas as pd, matplotlib.pyplot as plt
from PIL import Image

# Load metadata
df = pd.read_csv('/home/jovyan/work/rare-disease-cvae/data/raw/ISIC_2019_Training_GroundTruth.csv')
print(df.shape)
print(df.head())

# Class distribution
class_counts = df.iloc[:, 1:].sum().sort_values()
class_counts.plot(kind='barh', figsize=(8, 5), title='Images per class')
plt.tight_layout()
plt.savefig('/home/jovyan/work/rare-disease-cvae/outputs/class_distribution.png')
plt.show()

# Sample images from rare classes
rare_classes = ['DF', 'VASC']   # Dermatofibroma, Vascular — fewest samples
img_dir = '/home/jovyan/work/rare-disease-cvae/data/raw/ISIC_2019_Training_Input/ISIC_2019_Training_Input'

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for row, cls in enumerate(rare_classes):
    img_ids = df[df[cls] == 1]['image'].values[:5]
    for col, img_id in enumerate(img_ids):
        img = Image.open(f'{img_dir}/{img_id}.jpg')
        axes[row][col].imshow(img)
        axes[row][col].set_title(f'{cls}', fontsize=9)
        axes[row][col].axis('off')
plt.tight_layout()
plt.savefig('/home/jovyan/work/rare-disease-cvae/outputs/sample_images.png')
plt.show()

print("Rare class counts:")
print(class_counts[['DF', 'VASC']])