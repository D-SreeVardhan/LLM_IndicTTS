# Utterances Dataset

Spoken versions of the eight intents in `../multilingual_qa.json`, one folder per language.
`eval/multilingual_eval.py` generates them with the macOS system voices (Rishi for English,
Lekha for Hindi, Soumya for Kannada, Vani for Tamil) and resamples them to 16 kHz mono.

## Directory structure

```
utterances/
  en/  hi/  kn/  ta/
    m01.wav    # 16 kHz mono WAV
    m01.txt    # reference transcript (UTF-8), the question as written
    ...
    m08.wav
    m08.txt
```

## Adding real recordings

Any `<id>.wav` + `<id>.txt` pair placed in `hi/`, `kn/` or `ta/` is scored by `eval/wer_eval.py`.
Recordings of real students reading the questions are the most useful addition (see the paper's
limitations). Convert them first:

```bash
ffmpeg -i input.m4a -ar 16000 -ac 1 output.wav
```

## Running the evals

```bash
python eval/multilingual_eval.py            # regenerate audio if missing, then ASR + routing
python eval/wer_eval.py --decode both       # CTC vs RNNT word / character error rates
```
