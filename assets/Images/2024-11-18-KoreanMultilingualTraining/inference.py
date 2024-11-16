# This code is from https://github.com/facebookresearch/fairseq/blob/main/examples/textless_nlp/gslm/speech2unit/clustering/quantize_with_kmeans.py

import argparse
import tqdm
import joblib
import torch
import os
from glob import glob

from examples.textless_nlp.gslm.speech2unit.pretrained.hubert_feature_reader import (
    HubertFeatureReader,
)

from util import save_unit

def load_model(model_path, kmeans_path, use_cuda=False):
    hubert_reader = HubertFeatureReader(
        checkpoint_path=model_path,
        layer=11,
        use_cuda=use_cuda,
    )
    kmeans_model = joblib.load(open(kmeans_path, "rb"))
    kmeans_model.verbose = False

    return hubert_reader, kmeans_model

def main(args):
    use_cuda = torch.cuda.is_available() and not args.cpu

    hubert_reader, kmeans_model = load_model(args.mhubert_path, args.kmeans_path, use_cuda=use_cuda)

    # 입력 디렉토리에서 모든 WAV 파일 가져오기
    in_wav_paths = glob(os.path.join(args.in_wav_path, "*.wav"))
    # 출력 디렉토리에 저장될 파일 경로 생성
    out_unit_paths = [os.path.join(args.out_unit_path, os.path.splitext(os.path.basename(p))[0] + ".unit") for p in in_wav_paths]

    for in_wav_path, out_unit_path in tqdm.tqdm(
        zip(in_wav_paths, out_unit_paths),
        total=len(in_wav_paths)
    ):
        feats = hubert_reader.get_feats(in_wav_path)
        feats = feats.cpu().numpy()

        pred = kmeans_model.predict(feats)
        pred_str = " ".join(str(p) for p in pred)

        save_unit(pred_str, out_unit_path)

def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-wav-path", type=str, required=True, help="오디오 입력이 저장된 디렉토리 경로"
    )
    parser.add_argument(
        "--out-unit-path", type=str, required=True, help="유닛 출력이 저장될 디렉토리 경로"
    )
    parser.add_argument(
        "--mhubert-path",
        type=str,
        required=True,
        help="사전 학습된 mHuBERT 모델 체크포인트"
    )
    parser.add_argument(
        "--kmeans-path",
        type=str,
        required=True,
        help="추론에 사용할 K-means 모델 파일 경로",
    )
    parser.add_argument("--cpu", action="store_true", help="CPU에서 실행")
    args = parser.parse_args()
    main(args)

if __name__ == "__main__":
    cli_main()
