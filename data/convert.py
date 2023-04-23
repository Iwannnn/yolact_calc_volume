import cv2
import os.path
import glob
import numpy as np
from PIL import Image


def convertPNG(pgmfile, outdir):
    # 读取灰度图
    name = os.path.basename(pcdfile)
    png_name = ".".join(name.split(".")[:-1]) + ".png"
    me_name = ".".join(name.split(".")[:-1]) + "m.png"
    im_depth = cv2.imread(pgmfile, cv2.IMREAD_UNCHANGED)
    median_img = cv2.medianBlur(im_depth, 5)
    # 转换成伪彩色之前必须是8位图片）
    # 这里有个alpha值，深度图转换伪彩色图的scale可以通过alpha的数值调整，我设置为1，感觉对比度大一些
    im_color = cv2.applyColorMap(
        cv2.convertScaleAbs(im_depth, alpha=0.1), cv2.COLORMAP_JET
    )
    me_color = cv2.applyColorMap(
        cv2.convertScaleAbs(median_img, alpha=0.1), cv2.COLORMAP_JET
    )
    # 转成png
    im = Image.fromarray(im_color)
    me = Image.fromarray(me_color)
    # 保存图片
    im.save(os.path.join(outdir, os.path.basename(png_name)))
    me.save(os.path.join(outdir, os.path.basename(me_name)))


for pcdfile in glob.glob("E:\deeplearning\yolact\data\goods_input\*_d.pgm"):
    convertPNG(pcdfile, "E:\deeplearning\yolact\data\goods_outputs")
