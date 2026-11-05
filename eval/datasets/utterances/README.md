# Utterances Dataset

Place audio recordings here for WER evaluation.

## Directory structure

```
utterances/
  hi/
    001.wav    # 16 kHz mono WAV
    001.txt    # matching reference transcript (UTF-8)
    002.wav
    002.txt
    ...
  kn/
    001.wav
    001.txt
    ...
  ta/
    001.wav
    001.txt
    ...
```

## Sources

Collect at least 30 utterances per language covering campus vocabulary:

1. **Kathbath** (standardised Indian ASR benchmark)
   - https://github.com/AI4Bharat/Kathbath
   - Use test splits for hi, kn, ta

2. **IndicVoices**
   - https://huggingface.co/datasets/ai4bharat/IndicVoices
   - Filter by language; resample to 16 kHz mono

3. **Team-recorded RVU questions** (strongly recommended)
   - Record team members reading the Hindi/Kannada/Tamil translations of the
     gold_qa.json questions (converted using IndicTrans2)
   - This tests campus vocabulary: "CIE", "SEE", "hall ticket", "USN", "CGPA"

## Conversion to 16 kHz mono WAV

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

## Running the eval

```bash
python eval/wer_eval.py --decode both --lang all
```
