import scipy
import os
import matplotlib.pyplot as plt

def fuse_as(Aurora_dir, Star_dir):
    Aurora_imgs = []
    Star_imgs = []
    for aurora in os.listdir(Aurora_dir):
        imgs = scipy.io.loadmat(aurora)
        Aurora_imgs.append(imgs)
    for stars in os.listdir(Star_dir):
        imgs = scipy.io.loadmat(stars)
        Star_imgs.append(imgs)
