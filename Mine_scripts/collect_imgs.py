from icrawler.builtin import GoogleImageCrawler
from PIL import Image
import os
'''
This script is to collect certain types of image with keyword
It then saves in a provided output direction
'''

save_dir = "acropolis_images"
os.makedirs(save_dir, exist_ok=True)

crawler = GoogleImageCrawler(storage={'root_dir': save_dir})
crawler.crawl(keyword='Acropolis Athens', max_num=1000,filters={'type': 'photo'})

# resize to a fixed dimension, e.g. 512×512
for file in os.listdir(save_dir):
    path = os.path.join(save_dir, file)
    try:
        img = Image.open(path).convert("RGB")
        img = img.resize((512, 512))
        img.save(path)
    except Exception:
        os.remove(path)
