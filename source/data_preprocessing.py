import pandas as pd
import pytesseract as pt
import pdf2image 
import PIL.Image
import re

myconfig = r"==psm 6 --oem 3"

def read_text_from_image(path='datasets/CVs/Maksym-TautkevychiusCV-AI.pdf') -> str:
    
    images = convert_pdf(path)
    text = ""
    for image in images:
        text += pt.image_to_string(image, config=myconfig)
    clean_text = process_data(text)
    return clean_text

def process_data(text) -> str:
    
    clean_text = re.sub(r"[;)-/,.!@#$%^&*(\\]", "", text)
    return clean_text

def convert_pdf(path):

    images = pdf2image.convert_from_path(path)
    return images

