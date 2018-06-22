# -*- coding: utf-8 -*-
"""
Created on Wed May 16 11:37:34 2018

@author: deepe
"""
from PIL import Image
from PIL.ImageOps import grayscale

image = Image.open('sample.jpg')
image = grayscale(image).rotate(90)
half_the_width = image.size[0] / 2
half_the_height = image.size[1] / 2
newimg = image.crop(
        (
        half_the_width - 80,
        half_the_height - 102,
        half_the_width + 80,
        half_the_height + 102)
)
newimg.thumbnail((75,75))
newimg.show()

newimg.save("Image1.png")