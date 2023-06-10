#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import argparse

import cv2

from cvdrawtext import CvDrawText
from ppocr_onnx.ppocr_onnx import PaddleOcrONNX


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--movie", type=str, default=None)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--width", help='cap width', type=int, default=640)
    parser.add_argument("--height", help='cap height', type=int, default=360)

    parser.add_argument(
        "--det_model",
        type=str,
        default='./ppocr_onnx/model/det_model/ch_PP-OCRv3_det_infer.onnx',
    )
    parser.add_argument(
        "--rec_model",
        type=str,
        default='./ppocr_onnx/model/rec_model/japan_PP-OCRv3_rec_infer.onnx',
    )
    parser.add_argument(
        "--rec_char_dict",
        type=str,
        default='./ppocr_onnx/ppocr/utils/dict/japan_dict.txt',
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
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    if args.movie is not None:
        cap_device = args.movie
    image_path = args.image

    # PaddleOCR準備
    paddleocr_parameter = get_paddleocr_parameter()

    paddleocr_parameter.det_model_dir = args.det_model
    paddleocr_parameter.rec_model_dir = args.rec_model
    paddleocr_parameter.rec_char_dict_path = args.rec_char_dict
    paddleocr_parameter.cls_model_dir = args.cls_model

    paddleocr_parameter.use_gpu = args.use_gpu

    paddle_ocr_onnx = PaddleOcrONNX(paddleocr_parameter)

    # カメラ準備
    if image_path is None:
        cap = cv2.VideoCapture(cap_device)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)

    if image_path is not None:
        image = cv2.imread(image_path)

        # OCR実施
        dt_boxes, rec_res, time_dict = paddle_ocr_onnx(image)

        # デバッグ描画
        image = draw_info(image, dt_boxes, rec_res, time_dict)

        cv2.imshow('PaddleOCR ONNX Sample', image)
        _ = cv2.waitKey(-1)
    else:
        while True:
            # カメラキャプチャ
            ret, frame = cap.read()
            if not ret:
                break
            debug_image = copy.deepcopy(frame)

            # OCR実施
            dt_boxes, rec_res, time_dict = paddle_ocr_onnx(frame)

            # デバッグ描画
            debug_image = draw_info(debug_image, dt_boxes, rec_res, time_dict)

            # キー処理(ESC：終了)
            key = cv2.waitKey(1)
            if key == 27:  # ESC
                break

            cv2.imshow('PaddleOCR ONNX Sample', debug_image)
        cap.release()
        cv2.destroyAllWindows()


def draw_info(image, dt_boxes, rec_res, time_dict):
    for bbox, text in zip(dt_boxes, rec_res):
        cv2.line(
            image,
            pt1=(int(bbox[0][0]), int(bbox[0][1])),
            pt2=(int(bbox[1][0]), int(bbox[1][1])),
            color=(0, 255, 0),
            thickness=1,
            lineType=cv2.LINE_4,
        )
        cv2.line(
            image,
            pt1=(int(bbox[1][0]), int(bbox[1][1])),
            pt2=(int(bbox[2][0]), int(bbox[2][1])),
            color=(0, 255, 0),
            thickness=1,
            lineType=cv2.LINE_4,
        )
        cv2.line(
            image,
            pt1=(int(bbox[2][0]), int(bbox[2][1])),
            pt2=(int(bbox[3][0]), int(bbox[3][1])),
            color=(0, 255, 0),
            thickness=1,
            lineType=cv2.LINE_4,
        )
        cv2.line(
            image,
            pt1=(int(bbox[3][0]), int(bbox[3][1])),
            pt2=(int(bbox[0][0]), int(bbox[0][1])),
            color=(0, 255, 0),
            thickness=1,
            lineType=cv2.LINE_4,
        )

        font_path = './cvdrawtext/font/LINE_Seed_JP/Desktop/OTF/LINESeedJP_OTF_Rg.otf'
        image = CvDrawText.puttext(
            image,
            text[0],
            (int(bbox[0][0]), int(bbox[0][1])),
            font_path,
            20,
            (0, 255, 0),
        )

    # 推論時間
    text = 'Elapsed time(Det):' + '%.2f' % (time_dict['det'])
    text = text + 's'
    cv2.putText(
        image,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        thickness=2,
    )
    text = 'Elapsed time(Rec):' + '%.2f' % (time_dict['rec'])
    text = text + 's'
    cv2.putText(
        image,
        text,
        (10, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        thickness=2,
    )
    text = 'Elapsed time(All):' + '%.2f' % (time_dict['all'])
    text = text + 's'
    cv2.putText(
        image,
        text,
        (10, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        thickness=2,
    )
    return image


if __name__ == '__main__':
    main()
