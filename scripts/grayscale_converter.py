from PIL import Image
import os

src = "sketches/"
dst = "sketches_grayscale/"
os.makedirs(dst, exist_ok=True)

for f in os.listdir(src):
    if f.endswith(".png"):
        img = Image.open(os.path.join(src, f)).convert("L")
        img.save(os.path.join(dst, f))
