#!/usr/bin/env bash
# Second fetch queue: commercially-clean Piper English voices, kokoro int8, espeak-ng MSI, Fastspeech2_HS Tamil.
M=C:/tts/ttsPIPE/data/models
HF=https://huggingface.co/rhasspy/piper-voices/resolve/main
get(){ mkdir -p "$(dirname "$2")"; curl -sfL --retry 5 -o "$2" "$1"; echo "done $(stat -c %s "$2") $2"; }
for v in en/en_US/kristin/medium/en_US-kristin-medium en/en_US/ljspeech/medium/en_US-ljspeech-medium; do b=$(basename $v); d=$(dirname $v)
  get $HF/$v.onnx.json $M/piper/$b.onnx.json; get $HF/$d/MODEL_CARD $M/piper/$b.MODEL_CARD; get $HF/$v.onnx $M/piper/$b.onnx; done
get https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/kokoro-v1.0.int8.onnx $M/kokoro/kokoro-v1.0.int8.onnx
get https://github.com/espeak-ng/espeak-ng/releases/download/1.52.0/espeak-ng.msi $M/espeak-ng/espeak-ng.msi
G=https://media.githubusercontent.com/media/smtiitm/Fastspeech2_HS/main
for f in tamil/female/model/model.pth; do get $G/$f $M/fs2hs/$f; done
R=https://raw.githubusercontent.com/smtiitm/Fastspeech2_HS/main
for f in tamil/female/model/config.yaml tamil/female/model/energy_stats.npz tamil/female/model/feats_stats.npz tamil/female/model/feats_type tamil/female/model/pitch_stats.npz vocoder/female/dravidian/hifigan/config.json vocoder/female/dravidian/hifigan/generator; do get $R/$f $M/fs2hs/$f; done
echo ALL_DONE2
