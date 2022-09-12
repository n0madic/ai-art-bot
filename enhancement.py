from PIL import Image
import cv2
import numpy
import os


upscaling = os.getenv('UPSCALING', 'true').lower() == 'true'
if upscaling:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan.utils import RealESRGANer
    realesrgan_model_path = 'realesrgan/RealESRGAN_x4plus.pth'
    realesrgan_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(scale=4, model_path=realesrgan_model_path, model=realesrgan_model)

face_restoring = os.getenv('FACE_RESTORING', 'false').lower() == 'true'
if face_restoring:
    from gfpgan.utils import GFPGANer
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    face_enhancer = GFPGANer(model_path='gfpgan/GFPGANv1.3.pth', upscale=4, arch='clean', channel_multiplier=2, bg_upsampler=upsampler)


def face_presence_detection(image):
    if face_restoring:
        image = numpy.array(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)
        return len(faces) > 0
    else:
        return False


def upscale(image, face_restore=False):
    image = numpy.array(image)
    if face_restoring and face_restore:
        _, _, image = face_enhancer.enhance(image)
    else:
        image = upsampler.enhance(image, outscale=4)[0]
    return Image.fromarray(image)
