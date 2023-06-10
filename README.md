# PaddleOCR-ONNX-Sample
<img src="https://github.com/Kazuhito00/PaddleOCR-ONNX-Sample/assets/37477845/190adbc9-c65d-47b8-a7fe-d73f4f28e1b2" width="65%"><br>
[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)のPythonでのONNX推論サンプルです。<br>
[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)の推論スクリプトからPaddle依存処理とインポートを削除しています。<br>

# Model
以下リストのモデルを変換しています（蒸留モデルではないモデルを使用）<br>
* [OCR Model List（V3, updated on 2022.4.28）](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_en/models_list_en.md)

変換自体を試したい方はColaboratoryなどで[PaddleOCR-Convert2ONNX.ipynb](PaddleOCR-Convert2ONNX.ipynb)を使用ください。<br>

# Requirement
requirements.txt を参照ください。

# Demo
デモ(シンプルなアルファベット検出)の実行方法は以下です。
```bash
python demo_simple_ocr_en.py --image=sample.jpg
```
* --image<br>
OCR対象画像の指定<br>
デフォルト：sample.jpg
* --det_model<br>
テキスト検出モデルの指定<br>
デフォルト：./ppocr_onnx/model/det_model/en_PP-OCRv3_det_infer.onnx
* --rec_model<br>
テキスト認識モデルの指定<br>
デフォルト：./ppocr_onnx/model/rec_model/en_PP-OCRv3_rec_infer.onnx
* --rec_char_dict<br>
辞書データの指定<br>
デフォルト：./ppocr_onnx/ppocr/utils/dict/en_dict.txt
* --cls_model<br>
テキスト回転分類モデルの指定<br>
デフォルト：./ppocr_onnx/model/cls_model/ch_ppocr_mobile_v2.0_cls_infer.onnx
* --use_gpu<br>
GPU推論の利用<br>
デフォルト：指定なし


デモ(日本語検出＋表示)の実行方法は以下です。
```bash
python demo_draw_ocr_ja.py --image=sample.jpg
```
* --device<br>
カメラデバイス番号の指定<br>
デフォルト：0
* --movie<br>
動画ファイルの指定 ※指定時はカメラデバイスより優先<br>
デフォルト：指定なし
* --image<br>
画像ファイルの指定 ※指定時はカメラデバイスや動画より優先<br>
デフォルト：指定なし
* --width<br>
カメラキャプチャ時の横幅<br>
デフォルト：640
* --height<br>
カメラキャプチャ時の縦幅<br>
デフォルト：360
* --det_model<br>
テキスト検出モデルの指定<br>
デフォルト：./ppocr_onnx/model/det_model/en_PP-OCRv3_det_infer.onnx
* --rec_model<br>
テキスト認識モデルの指定<br>
デフォルト：./ppocr_onnx/model/rec_model/en_PP-OCRv3_rec_infer.onnx
* --rec_char_dict<br>
辞書データの指定<br>
デフォルト：./ppocr_onnx/ppocr/utils/dict/en_dict.txt
* --cls_model<br>
テキスト回転分類モデルの指定<br>
デフォルト：./ppocr_onnx/model/cls_model/ch_ppocr_mobile_v2.0_cls_infer.onnx
* --use_gpu<br>
GPU推論の利用<br>
デフォルト：指定なし

# Reference
* [PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)

# Author
高橋かずひと(https://twitter.com/KzhtTkhs)
 
# License 
PaddleOCR-ONNX-Sample is under [Apache2.0 License](LICENSE).

# License(Image)
サンプル画像は[フリー素材ぱくたそ](https://www.pakutaso.com)様の写真を利用しています。

# License(Font)
日本語フォントは[LINE Seed JP](https://seed.line.me/index_jp.html)を利用しています。
