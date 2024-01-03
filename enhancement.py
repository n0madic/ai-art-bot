from basicsr.archs.rrdbnet_arch import RRDBNet
from gfpgan.utils import GFPGANer
from realesrgan.utils import RealESRGANer
from PIL import Image
import numpy
import os


class Enhancement():
    def __init__(self, face_enhancer_model_path, face_enhancer_arch, realesrgan_model_path):
        self.face_enhancer_model_path = face_enhancer_model_path
        self.face_enhancer_arch = face_enhancer_arch
        self.realesrgan_model_path = realesrgan_model_path

    def fixface(self, image):
        if type(image) == str and os.path.exists(image):
            image = Image.open(image)
        info = image.info
        image = numpy.array(image)
        face_enhancer_no_scale = GFPGANer(model_path=self.face_enhancer_model_path, upscale=1, arch=self.face_enhancer_arch)
        _, _, image = face_enhancer_no_scale.enhance(image, weight=0.2)
        image = Image.fromarray(image)
        image.info = info
        return image


    def upscale(self, image, face_restore=False):
        info = image.info
        image = numpy.array(image)
        if face_restore:
            face_enhancer = GFPGANer(model_path=self.face_enhancer_model_path, upscale=4, arch=self.face_enhancer_arch, bg_upsampler=upsampler)
            _, _, image = face_enhancer.enhance(image)
        else:
            realesrgan_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            upsampler = RealESRGANer(scale=4, model_path=self.realesrgan_model_path, model=realesrgan_model)
            image = upsampler.enhance(image, outscale=4)[0]
        image = Image.fromarray(image)
        image.info = info
        return image
