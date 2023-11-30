import os
import xlwt
import shutil
import cv2
import sys
import math
import numpy as np
import itertools
import matplotlib.pyplot as plt
from aes import decrypt_aes,encrypt_aes
from PIL import Image
from pathlib import Path
#from scipy import signal
quant = np.array([[16,11,10,16,24,40,51,61],      # QUANTIZATION TABLE
                    [12,12,14,19,26,58,60,55],    # required for DCT
                    [14,13,16,24,40,57,69,56],
                    [14,17,22,29,51,87,80,62],
                    [18,22,37,56,68,109,103,77],
                    [24,35,55,64,81,104,113,92],
                    [49,64,78,87,103,121,120,101],
                    [72,92,95,98,112,100,103,99]])
'''def show(im):
    im_resized = cv2.resize(im, (500, 500), interpolation=cv2.INTER_LINEAR)
    plt.imshow(cv2.cvtColor(im_resized, cv2.COLOR_BGR2RGB))
    plt.show()'''
        


class LSB():
    #encoding part :
    def encode_image(self,img, msg):
        length = len(msg)
        if length > 255:
            print("text too long! (don't exeed 255 characters)")
            return False
        encoded = img.copy()
        width, height = img.size
        index = 0
        for row in range(height):
            for col in range(width):
                if img.mode != 'RGB':
                    r, g, b ,a = img.getpixel((col, row))
                elif img.mode == 'RGB':
                    r, g, b = img.getpixel((col, row))
                # first value is length of msg
                if row == 0 and col == 0 and index < length:
                    asc = length
                elif index <= length:
                    c = msg[index -1]
                    asc = ord(c)
                else:
                    asc = b
                encoded.putpixel((col, row), (r, g , asc))
                index += 1
        return encoded

    #decoding part :
    def decode_image(self,img):
        width, height = img.size
        msg = ""
        index = 0
        for row in range(height):
            for col in range(width):
                if img.mode != 'RGB':
                    r, g, b ,a = img.getpixel((col, row))
                elif img.mode == 'RGB':
                    r, g, b = img.getpixel((col, row))  
                # first pixel r value is length of message
                if row == 0 and col == 0:
                    length = b
                elif index <= length:
                    msg += chr(b)
                index += 1
        lsb_decoded_image_file = "lsb_" + original_image_file
        #img.save(lsb_decoded_image_file)
        ##print("Decoded image was saved!")
        return msg

class Compare():
    def correlation(self, img1, img2):
        return signal.correlate2d (img1, img2)
    def meanSquareError(self, img1, img2):
        error = np.sum((img1.astype('float') - img2.astype('float')) ** 2)
        error /= float(img1.shape[0] * img1.shape[1]);
        return error
    def psnr(self, img1, img2):
        mse = self.meanSquareError(img1,img2)
        if mse == 0:
            return 100
        PIXEL_MAX = 255.0
        return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))



#driver part :
#deleting previous folders :
if os.path.exists("Encoded_image/"):
    shutil.rmtree("Encoded_image/")
if os.path.exists("Decoded_output/"):
    shutil.rmtree("Decoded_output/")
if os.path.exists("Comparison_result/"):
    shutil.rmtree("Comparison_result/")
#creating new folders :
os.makedirs("Encoded_image/")
os.makedirs("Decoded_output/")
os.makedirs("Comparison_result/")
original_image_file = ""    # to make the file name global variable
lsb_encoded_image_file = ""
dct_encoded_image_file = ""
dwt_encoded_image_file = ""
key_encoded = ""

while True:
    # m = input("To encode press '1', to decode press '2', press any other button to close: ")
    m = input("Nhấn 1 để dấu tin trong ảnh, nhấn 2 để tách thông điệp ra khỏi ảnh, nhấn nút bất kỳ để đóng chương trình:")
    if m == "1":
        os.chdir("Original_image/")
        original_image_file = input("Nhập tên file : ")
        lsb_img = Image.open(original_image_file)
        print("Miêu tả : ",lsb_img,"\nMode : ", lsb_img.mode)
        secret_msg = input("Nhập thông điệp muốn dấu: ")
        print("Độ dài thông điệp: ",len(secret_msg))
        os.chdir("..")
        os.chdir("Encoded_image/")
        code_text = str(encrypt_aes(b'ThisIsA16ByteKey',secret_msg.encode('utf-8')))
        lsb_img_encoded = LSB().encode_image(lsb_img, code_text)
        lsb_encoded_image_file = "lsb_" + original_image_file
        lsb_img_encoded.save(lsb_encoded_image_file)
        print("Ảnh dấu tin đã được lưu!")
        os.chdir("..")

    elif m == "2":
        os.chdir("Encoded_image/")
        path_file_image_lsb = input('Nhập tên file:')
        lsb_img = Image.open(path_file_image_lsb)
        os.chdir("..") #going back to parent directory
        os.chdir("Decoded_output/")
        lsb_hidden_text = LSB().decode_image(lsb_img)
        print(lsb_hidden_text)
        file = open("lsb_hidden_text.txt","w")
        file.write(str(decrypt_aes(b'ThisIsA16ByteKey',lsb_hidden_text))) # saving hidden text as text file
        file.close()
        print("Thông điệp đã được lưu vào file!")
        os.chdir("..")
    else:
        print("Closed!")
        break