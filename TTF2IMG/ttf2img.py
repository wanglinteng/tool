import os
import pygame
import numpy as np


class Ttf2Image():

    def __init__(self, font_path, image_path, image_size, fmt):
        if not os.path.exists(image_path):
            os.makedirs(image_path)
        self.font_path = font_path
        self.image_path = image_path
        self.image_size = image_size
        self.fmt = fmt

    def extract_image(self, start, end):
        pygame.init()
        img_num = 0
        for codepoint in range(int(start), int(end) + 1):
            word = chr(codepoint)

            font = pygame.font.Font(self.font_path, self.image_size)
            text = font.render(word, True, (0, 0, 0), (255, 255, 255))
            if np.sum(pygame.PixelArray(text)) != 0:  # delete the word not in this ttf.
                screen = pygame.display.set_mode((self.image_size, self.image_size), 0, 32)
                background = pygame.Surface(screen.get_size())
                background.fill(color=(255, 255, 255))
                center = (background.get_width() / 2, background.get_height() / 2)
                text_pos = text.get_rect(center=center)
                background.blit(text, text_pos)
                screen.blit(background, (0, 0))

                pygame.image.save(screen, os.path.join(self.image_path, str(codepoint) + "." + self.fmt))
                img_num += 1
        print("Extract %d characters in this file!" % img_num)


if __name__ == '__main__':

    """
     install: pip install pygame numpy
    
     start, end = (0x4E00, 0x9FA5) # 汉字编码范围
     start, end = (0x30, 0x39) # 数字
     start, end = (0x61, 0x7A) # 小写字母
     start, end = (0x41, 0x5A) # 大写字母
    """
    font_path = "/opt/paperSegment/dataSet/fronts/print/GB2312.ttf"
    image_path = "./img"
    image_size = 128
    fmt = "jpg"
    worker = Ttf2Image(font_path, image_path, image_size, fmt)
    worker.extract_image(0x4E00, 0x9FA5)