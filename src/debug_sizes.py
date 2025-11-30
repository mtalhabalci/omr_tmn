import os
from PIL import Image
root = os.path.dirname(os.path.dirname(__file__))
page = os.path.join(root, 'ds2_dense_tmn', 'images', 'lg-101766503886095953-aug-beethoven--page-1.png')
seg = os.path.join(root, 'ds2_dense_tmn', 'segmentation', 'lg-101766503886095953-aug-beethoven--page-1_seg.png')
inst = os.path.join(root, 'ds2_dense_tmn', 'instance', 'lg-101766503886095953-aug-beethoven--page-1_inst.png')
pi = Image.open(page)
si = Image.open(seg)
ii = Image.open(inst)
print({'image': pi.mode, 'size': pi.size})
print({'seg': si.mode, 'size': si.size})
print({'inst': ii.mode, 'size': ii.size})
