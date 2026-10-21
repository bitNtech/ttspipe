#!/usr/bin/env bash
# Sequential, resumable model fetch (slow link: one stream at a time, priority order).
# Mirrors used for gated AI4Bharat repos are sha256-identical to upstream (verified via HF API lfs.sha256).
M=C:/tts/ttsPIPE/data/models
HF=https://huggingface.co
get(){ mkdir -p "$(dirname "$2")"; curl -sfL -C - --retry 5 -o "$2" "$1" || curl -sfL --retry 5 -o "$2" "$1"; echo "done $(stat -c %s "$2") $2"; }
for f in config.json tokenizer_config.json vocab.json special_tokens_map.json README.md model.safetensors; do get $HF/facebook/mms-tts-tam/resolve/main/$f $M/mms-tts-tam/$f; done
for v in ta_IN-rasa_female-medium ta_IN-rasa_male-medium; do get $HF/tinisoft/piper-$v/resolve/main/$v.onnx.json $M/piper/$v.onnx.json; get $HF/tinisoft/piper-$v/resolve/main/$v.onnx $M/piper/$v.onnx; done
K=https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1
for f in voices-v1.0.bin kokoro-v1.0.int8.onnx kokoro-v1.0.onnx; do get $K/$f $M/kokoro/$f; done
for v in en/en_US/lessac/medium/en_US-lessac-medium en/en_US/amy/medium/en_US-amy-medium; do b=$(basename $v); d=$(dirname $v)
  get $HF/rhasspy/piper-voices/resolve/main/$v.onnx.json $M/piper/$b.onnx.json; get $HF/rhasspy/piper-voices/resolve/main/$d/MODEL_CARD $M/piper/$b.MODEL_CARD; get $HF/rhasspy/piper-voices/resolve/main/$v.onnx $M/piper/$b.onnx; done
R=$HF/rsolanki1822/IndicF5-mirror/resolve/main
for f in config.json model.py checkpoints/vocab.txt prompts/MAR_F_HAPPY_00001.wav prompts/PAN_F_HAPPY_00001.wav model.safetensors; do get $R/$f $M/indicf5/$f; done
get https://github.com/AI4Bharat/Indic-TTS/releases/download/v1-checkpoints-release/ta.zip $M/indic-tts/ta.zip
for f in config.json generation_config.json preprocessor_config.json special_tokens_map.json tokenizer.json tokenizer.model tokenizer_config.json README.md model.safetensors; do get $HF/RXD03/indic-parler-tts/resolve/main/$f $M/indic-parler-tts/$f; done
get $K/kokoro-v1.0.fp16.onnx $M/kokoro/kokoro-v1.0.fp16.onnx
echo ALL_DONE
