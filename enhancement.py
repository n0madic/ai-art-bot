from basicsr.archs.rrdbnet_arch import RRDBNet
from gfpgan.utils import GFPGANer
from realesrgan.utils import RealESRGANer
from PIL import Image
import config
import cv2
import numpy
import os
import torch


def face_presence_detection(image):
    if config.cfg.face_restoring:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        image = numpy.array(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)
        return len(faces) > 0
    else:
        return False


def fixface(image):
    if type(image) == str and os.path.exists(image):
        image = Image.open(image)
    info = image.info
    image = numpy.array(image)
    face_enhancer_no_scale = GFPGANer(model_path=config.cfg.face_enhancer_model_path, upscale=1, arch=config.cfg.face_enhancer_arch)
    _, _, image = face_enhancer_no_scale.enhance(image, weight=0.2)
    image = Image.fromarray(image)
    image.info = info
    return image


def upscale(image, face_restore=False):
    info = image.info
    image = numpy.array(image)
    if face_restore:
        face_enhancer = GFPGANer(model_path=config.cfg.face_enhancer_model_path, upscale=4, arch=config.cfg.face_enhancer_arch, bg_upsampler=upsampler)
        _, _, image = face_enhancer.enhance(image)
    else:
        realesrgan_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        upsampler = RealESRGANer(scale=4, model_path=config.cfg.realesrgan_model_path, model=realesrgan_model)
        image = upsampler.enhance(image, outscale=4)[0]
    image = Image.fromarray(image)
    image.info = info
    return image
