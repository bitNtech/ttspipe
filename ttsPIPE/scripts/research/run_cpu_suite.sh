#!/usr/bin/env bash
# Re-runnable suite for the onnxruntime / C engines (main venv).
cd "$(dirname "$0")"; PY=../../.venv-research/Scripts/python.exe; export PYTHONIOENCODING=utf-8
$PY bench_espeak.py --threads 1
for t in 4 8; do
  for v in ta_IN-rasa_female-medium ta_IN-rasa_male-medium en_US-lessac-medium en_US-amy-medium; do $PY bench_piper.py --tag $v --threads $t; done
  $PY bench_kokoro.py --tag fp32 --threads $t
done
