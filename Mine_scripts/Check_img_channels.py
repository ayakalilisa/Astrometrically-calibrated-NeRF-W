from PIL import Image
import numpy as np

img_check = Image.open('/Users/ayaka/Documents/UiT/Master/nerf_mine_scripts/Nerftest_img/IMG_9650.jpg')
print(f'Check tea cup image channels:{img_check.mode} and shape:{np.array(img_check).shape}')

