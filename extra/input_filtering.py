#!/usr/bin/env python3

from PIL import Image, ImageFilter

if __name__ == "__main__":
    im = Image.open("../dataset/benign/benign (92).png").convert("L")
    im.filter(ImageFilter.MedianFilter(size=21)).save("test.png")
