#!/usr/bin/env bash
# Clean, sequential re-measurement (nothing else running on the box). 1 warm-up + 5 timed runs per text.
cd "$(dirname "$0")"; PY=../../.venv-research/Scripts/python.exe; export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
$PY bench_espeak.py --threads 1
for t in 1 2 4; do $PY bench_piper.py --tag ta_IN-rasa_female-medium --threads $t; done
$PY bench_piper.py --tag ta_IN-rasa_male-medium --threads 2
for v in en_US-ljspeech-medium en_US-kristin-medium en_US-lessac-medium; do $PY bench_piper.py --tag $v --threads 2; done
$PY bench_piper.py --tag en_US-ljspeech-medium --threads 4
for t in 2 4 8; do $PY bench_kokoro.py --tag fp32 --threads $t; done
for t in 4; do $PY bench_kokoro.py --tag int8 --threads $t; done
for t in 2 4 8; do $PY bench_mms.py --threads $t; done
echo CLEAN_SUITE_DONE
