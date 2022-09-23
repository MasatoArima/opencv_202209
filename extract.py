# -*- coding: utf-8 -*-

import cv2
import os
import numpy as np


def img_write(name, img):
    # 画像を保存する
    cv2.imwrite(output_path + "/" + name + '.png', img)


def read_file(dire, name, cv2type=None):
    # 画像を読み込む
    file = (str(dire) + "\\" + str(name))
    if cv2type == None:
        file = cv2.imread(file)
    elif cv2type == 'IMREAD_GRAYSCALE':
        file = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    return file


def img_show(name, img):
    # 画像を表示する
    cv2.imshow(name, img)
    cv2.waitKey(0)


def make_mask(img):
    # マスク画像を作成する

    #### 設定パラメータ ########################################
    max_value = 10000  # 領域判定の上限値                      #
    min_value = 20  # 領域判定の下限値                         #
    add_index = 24  # 領域の追加設定(抽出した輪郭のIndexNo)    #
    ############################################################

    # マスク画像Rowデータ作成・inputデータ読込
    global grayscale_img
    global binary_image
    global contour_img
    campus = np.zeros((int(read_img.shape[0]), int(read_img.shape[1])), dtype=np.uint8)
    raw_campus = np.zeros_like(campus, dtype=np.uint8)

    # 2値画像作成
    grayscale_img = read_file(input_path, img, 'IMREAD_GRAYSCALE')
    ret, binary_image = cv2.threshold(grayscale_img, 0, 255, cv2.THRESH_OTSU)

    # 輪郭を抽出し, IndexNo 、面積を取得
    contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contour_area = {}
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area > min_value and area < max_value:
            contour_area[i] = area

    # 輪郭を抽出しマスクを作成
    if 'add_index' in locals():
        contour_img = cv2.drawContours(read_img, contours, add_index, (255, 0, 255), 2)
        contour_img = cv2.drawContours(read_img, contours, max(contour_area, key=contour_area.get), (255, 0, 255), 2)
        mask_img = cv2.drawContours(raw_campus, contours, add_index, color=255, thickness=-1)
        mask_img = cv2.drawContours(raw_campus, contours, max(contour_area, key=contour_area.get), color=255, thickness=-1)
    else:
        contour_img = cv2.drawContours(read_img, contours, max(contour_area, key=contour_area.get), (255, 0, 255), 2)
        mask_img = cv2.drawContours(raw_campus, contours, max(contour_area, key=contour_area.get), color=255, thickness=-1)

    # データ書込
    img_write('grayscale_image', grayscale_img)
    img_write('binary_image', binary_image)
    img_write('contour_image', contour_img)
    img_write('mask_image', mask_img)

    return grayscale_img, binary_image, contour_img, mask_img

# 処理開始＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿


####### 設定パラメータ #########################################################
input_path = os.getcwd() + '\\input_images'     # inputファイルの場所を設定    #
output_path = os.getcwd() + '\\output_images'   # outputファイルの場所を設定   #
input_file = 'milkdrop.bmp'                     # 処理するファイル名を設定     #
################################################################################

# オリジナルデータ (読込 / 書込)
read_img = read_file(input_path, input_file)
img_write('original_image', read_img)

# MASK作成#
make_mask(input_file)

# 領域抽出
original_img = read_file(output_path, 'original_image.png')
mask_img = read_file(output_path, 'mask_image.png')
dst = cv2.bitwise_and(original_img, mask_img)
img_write('extraction_image', dst)
img_show('compare_image ( original vs extraction )',cv2.hconcat([original_img, dst]))

# 各工程の画像
# read_img = read_file(input_path, input_file)
# img_show('original_image', read_img)
# img_show('grayscale_image', grayscale_img)
# img_show('binary_image', binary_image)
# img_show('contour_image', contour_img)
# img_show('mask_image', mask_img)
# img_show('extraction_img', dst)
