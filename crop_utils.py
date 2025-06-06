import numpy as np
from PIL import Image

class CropBlackBorders:
    """
    Transform que recorta los bordes completamente negros de una imagen PIL.
    Se basa en encontrar el bounding box de todos los píxeles > 0.
    """


    def __call__(self, img: Image.Image) -> Image.Image:
        # 1. Crear copia en escala de grises para detectar negro
        img_gray = img.convert("L")
        np_img = np.array(img_gray)

        # 2. Encuentra filas y columnas que contienen píxeles > 0
        non_zero_rows = np.where(np_img.max(axis=1) > 0)[0]
        non_zero_cols = np.where(np_img.max(axis=0) > 0)[0]

        # 3. Si toda la imagen es negra, devolvemos la imagen RGB tal cual
        if non_zero_rows.size == 0 or non_zero_cols.size == 0:
            return img

        # 4. Calcular coordenadas del bounding box
        top_row = non_zero_rows[0]
        bottom_row = non_zero_rows[-1]
        left_col = non_zero_cols[0]
        right_col = non_zero_cols[-1]

        # 5. Recortar la imagen original RGB
        cropped = img.crop((left_col, top_row, right_col + 1, bottom_row + 1))
        return cropped

    def __repr__(self):
        return f"{self.__class__.__name__}()"