#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse

import cv2

from ppocr_onnx.ppocr_onnx import PaddleOcrONNX


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image", type=str, default='image/sample.jpg')

    parser.add_argument(
        "--det_model",
        type=str,
        default='./ppocr_onnx/model/det_model/en_PP-OCRv3_det_infer.onnx',
    )
    parser.add_argument(
        "--rec_model",
        type=str,
        default='./ppocr_onnx/model/rec_model/en_PP-OCRv3_rec_infer.onnx',
    )
    parser.add_argument(
        "--rec_char_dict",
        type=str,
        default='./ppocr_onnx/ppocr/utils/dict/en_dict.txt',
    )
    parser.add_argument(
        "--cls_model",
        type=str,
        default=
        './ppocr_onnx/model/cls_model/ch_ppocr_mobile_v2.0_cls_infer.onnx',
    )

    parser.add_argument(
        "--use_gpu",
        action="store_true",
    )

    args = parser.parse_args()

    return args


class DictDotNotation(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def get_paddleocr_parameter():
    paddleocr_parameter = DictDotNotation()

    # params for prediction engine
    paddleocr_parameter.use_gpu = False

    # params for text detector
    paddleocr_parameter.det_algorithm = 'DB'
    paddleocr_parameter.det_model_dir = './ppocr_onnx/model/det_model/ch_PP-OCRv3_det_infer.onnx'
    paddleocr_parameter.det_limit_side_len = 960
    paddleocr_parameter.det_limit_type = 'max'
    paddleocr_parameter.det_box_type = 'quad'

    # DB parmas
    paddleocr_parameter.det_db_thresh = 0.3
    paddleocr_parameter.det_db_box_thresh = 0.6
    paddleocr_parameter.det_db_unclip_ratio = 1.5
    paddleocr_parameter.max_batch_size = 10
    paddleocr_parameter.use_dilation = False
    paddleocr_parameter.det_db_score_mode = 'fast'

    # params for text recognizer
    paddleocr_parameter.rec_algorithm = 'SVTR_LCNet'
    paddleocr_parameter.rec_model_dir = './ppocr_onnx/model/rec_model/japan_PP-OCRv3_rec_infer.onnx'
    paddleocr_parameter.rec_image_shape = '3, 48, 320'
    paddleocr_parameter.rec_batch_num = 6
    paddleocr_parameter.rec_char_dict_path = './ppocr_onnx/ppocr/utils/dict/japan_dict.txt'
    paddleocr_parameter.use_space_char = True
    paddleocr_parameter.drop_score = 0.5

    # params for text classifier
    paddleocr_parameter.use_angle_cls = False
    paddleocr_parameter.cls_model_dir = './ppocr_onnx/model/cls_model/ch_ppocr_mobile_v2.0_cls_infer.onnx'
    paddleocr_parameter.cls_image_shape = '3, 48, 192'
    paddleocr_parameter.label_list = ['0', '180']
    paddleocr_parameter.cls_batch_num = 6
    paddleocr_parameter.cls_thresh = 0.9

    paddleocr_parameter.save_crop_res = False

    return paddleocr_parameter


def main():
    # コマンドライン引数
    args = get_args()
    image_path = args.image

    # PaddleOCR準備
    paddleocr_parameter = get_paddleocr_parameter()

    paddleocr_parameter.det_model_dir = args.det_model
    paddleocr_parameter.rec_model_dir = args.rec_model
    paddleocr_parameter.rec_char_dict_path = args.rec_char_dict
    paddleocr_parameter.cls_model_dir = args.cls_model

    paddleocr_parameter.use_gpu = args.use_gpu

    paddle_ocr_onnx = PaddleOcrONNX(paddleocr_parameter)

    # 画像読み込み
    image = cv2.imread(image_path)

    # OCR実施
    dt_boxes, rec_res, time_dict = paddle_ocr_onnx(image)

    print(time_dict)
    for dt_box, rec in zip(dt_boxes, rec_res):
        print(dt_box, rec)


if __name__ == '__main__':
    main()
