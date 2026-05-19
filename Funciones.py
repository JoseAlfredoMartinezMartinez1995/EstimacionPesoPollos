import numpy as np
from plantcv import plantcv as pcv
import io
import cv2
from IPython.display import clear_output
from sklearn.model_selection import GridSearchCV

def Preprocesamiento(imagen, pix):
    # Extraer el canal l
    img_l = pcv.rgb2gray_lab(rgb_img = imagen, channel = 'l')
    img_lb = pcv.threshold.otsu(gray_img = img_l, object_type = 'light')
    
    # Eliminar ruido
    img_mask01 = pcv.erode(gray_img = img_lb, ksize = 4, i = 1)
    img_mask01 = pcv.fill(bin_img = img_mask01, size = 1000)
    img_mask01 = pcv.fill_holes(bin_img = img_mask01)

    # Limpiar cualquier ruido en los bordes    
    alto = imagen.shape[0]
    largo = imagen.shape[1]
    
    roi1 = pcv.roi.rectangle(img = imagen, x = 0, y = 0, h = alto, w = pix)
    img_mask02 = pcv.roi.filter(mask = img_mask01, roi = roi1, roi_type = 'partial')
    
    roi2 = pcv.roi.rectangle(img = imagen, x = 0, y = 0, h = pix, w = largo)
    img_mask03 = pcv.roi.filter(mask = img_mask01, roi = roi2, roi_type = 'partial')
    
    roi3 = pcv.roi.rectangle(img = imagen, x = 0, y = alto - pix, h = pix, w = largo)
    img_mask04 = pcv.roi.filter(mask = img_mask01, roi = roi3, roi_type = 'partial')
    
    roi4 = pcv.roi.rectangle(img = imagen, x = largo - pix, y = 0, h = alto, w = pix)
    img_mask05 = pcv.roi.filter(mask = img_mask01, roi = roi4, roi_type = 'partial')
    
    img_maskX = pcv.logical_or(img_mask02, img_mask03)
    img_maskX = pcv.logical_or(img_maskX, img_mask04)
    img_maskX = pcv.logical_or(img_maskX, img_mask05)
    
    img_mask01 -= img_maskX

    imagen2 = pcv.apply_mask(img = imagen, mask = img_mask01, mask_color = 'white')

    # contar objetos
    etiquetas, n = pcv.create_labels(mask = img_mask01)

    return imagen2, etiquetas, n

def Preprocesamiento2(imagen):
    # Extraer el canal l
    img_l = pcv.rgb2gray_lab(rgb_img = imagen, channel = 'l')
    img_lb = pcv.threshold.otsu(gray_img = img_l, object_type = 'light')
    
    # Eliminar ruido
    img_mask01 = pcv.erode(gray_img = img_lb, ksize = 4, i = 1)
    img_mask01 = pcv.fill(bin_img = img_mask01, size = 1000)
    img_mask01 = pcv.fill_holes(bin_img = img_mask01)

    imagen2 = pcv.apply_mask(img = imagen, mask = img_mask01, mask_color = 'white')

    # contar objetos
    etiquetas, n = pcv.create_labels(mask = img_mask01)

    return imagen2, etiquetas, n