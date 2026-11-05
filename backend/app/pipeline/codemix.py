"""
codemix.py
──────────
IndicConformer transcribes sounds, not meaning: an English question spoken
with Hindi selected comes back as English written in Devanagari
("वेन इज द नेक्स्ट एग्जाम").  This module detects such transcripts and
rewrites them in the native language via an English round trip
("When is the next exam?" → "अगली परीक्षा कब होगी?").

Detection is deliberately conservative: the transcript must contain at least
one common English word spelled in the native script AND none of the common
native function / question words, so genuine native speech is never rewritten.
"""

import re

from loguru import logger

from backend.app.pipeline import asr, translate

# Common English words as IndicConformer spells them (whole-word match).
# Words that are also real native words (e.g. Hindi इस, इन, दी, आई) are excluded.
_ENGLISH_IN_SCRIPT: dict[str, set[str]] = {
    "hi": {
        "वेन", "वैन", "व्हेन", "वॉट", "व्हाट", "वाट", "व्हेयर", "वेयर", "हाउ", "हाऊ", "व्हाई", "वाई", "हू",
        "इज", "इज़", "आर", "वाज", "वाज़", "विल", "कैन", "कैंन", "डू", "डज", "डज़", "द", "दि",
        "ऑफ", "ऑफ़", "फॉर", "फ़ॉर", "ऑन", "एट", "माय", "यू", "योर", "एंड", "नेक्स्ट",
        "प्लीज", "प्लीज़", "टेल", "मी", "अबाउट", "दिस", "दैट", "देयर", "एनी", "आवर",
    },
    "kn": {
        "ವೆನ್", "ವ್ಹೆನ್", "ವಾಟ್", "ವ್ಹಾಟ್", "ವೇರ್", "ಹೌ", "ವೈ", "ಹೂ", "ಇಸ್", "ಈಸ್", "ಇಜ್", "ಆರ್",
        "ವಿಲ್", "ಕ್ಯಾನ್", "ಡು", "ದ", "ದಿ", "ಆಫ್", "ಫಾರ್", "ಆನ್", "ಮೈ", "ಯು", "ಯುವರ್", "ಅಂಡ್",
        "ನೆಕ್ಸ್ಟ್", "ಪ್ಲೀಸ್", "ಟೆಲ್", "ಮಿ", "ಅಬೌಟ್", "ದಿಸ್", "ದ್ಯಾಟ್",
    },
    "ta": {
        "வென்", "வாட்", "வேர்", "ஹவ்", "ஹௌ", "வை", "ஹூ", "இஸ்", "இஸ", "ஆர்", "வில்", "கேன்",
        "டு", "த", "தி", "ஆஃப்", "ஃபார்", "ஆன்", "மை", "யூ", "யுவர்", "அண்ட்", "நெக்ஸ்ட்",
        "ப்ளீஸ்", "டெல்", "மீ", "அபவுட்", "திஸ்", "தட்",
    },
}

# Native function / question words. Hindi is matched as whole words; Kannada
# and Tamil are agglutinative, so their stems are matched as substrings.
_NATIVE_MARKERS: dict[str, set[str]] = {
    "hi": {
        "है", "हैं", "का", "की", "के", "में", "से", "को", "कब", "क्या", "कैसे", "कितना", "कितनी",
        "कितने", "कौन", "कौनसा", "कहाँ", "कहां", "क्यों", "और", "या", "पर", "भी", "मुझे", "मैं",
        "हम", "आप", "यह", "वह", "ये", "वो", "था", "थी", "थे", "नहीं", "होगा", "होगी", "होंगे",
        "होता", "होती", "मिलेगा", "मिलेगी", "बताओ", "बताइए", "बताइये", "बताएं", "अगला", "अगली",
        "लिए", "वाला", "वाली", "परीक्षा",
    },
    "kn": {
        "ಯಾವಾಗ", "ಏನು", "ಎಷ್ಟು", "ಹೇಗೆ", "ಯಾರು", "ಎಲ್ಲಿ", "ಯಾವ", "ಇದೆ", "ಇವೆ", "ಬೇಕು", "ಮತ್ತು",
        "ಅಥವಾ", "ನನಗೆ", "ನಾನು", "ನನ್ನ", "ಮುಂದಿನ", "ಪರೀಕ್ಷೆ", "ಸಿಗುತ್ತ", "ಹೇಳಿ", "ಎಂದರೇನು",
    },
    "ta": {
        "எப்போது", "என்ன", "எவ்வளவு", "எப்படி", "யார்", "எங்கே", "எந்த", "உள்ளது", "உள்ளன",
        "இருக்கிற", "வேண்டும்", "மற்றும்", "அல்லது", "எனக்கு", "நான்", "அடுத்த", "தேர்வு",
        "கிடைக்கும்", "சொல்லுங்கள்", "என்றால்",
    },
}

_WORD_RE = re.compile(r"[^\s.,!?।]+")

# IndicConformer has no English model. English speech is recognised with the
# Hindi model (which writes it in Devanagari) and translated with IndicTrans2.
ENGLISH_ASR_PROXY = "hi"

# Question words the Hindi model often mishears at the start of an English question
_LEADING_FIXES = {"वैन": "व्हेन", "वेन": "व्हेन", "वॉट": "व्हाट", "वाट": "व्हाट"}


def looks_like_transliterated_english(text: str, lang: str) -> bool:
    english = _ENGLISH_IN_SCRIPT.get(lang)
    if not english:
        return False
    words = _WORD_RE.findall(text)
    if len(words) < 2 or not any(w in english for w in words):
        return False
    markers = _NATIVE_MARKERS[lang]
    if lang == "hi":
        return not any(w in markers for w in words)
    return not any(m in text for m in markers)


def _fix_leading_word(text: str) -> str:
    first, _, rest = text.strip().partition(" ")
    return f"{_LEADING_FIXES.get(first, first)} {rest}".strip()


def english_from_proxy_transcript(text: str) -> str:
    """English text for speech recognised by the Hindi proxy model."""
    if not text.strip():
        return text
    english, _ = translate.to_english(_fix_leading_word(text), ENGLISH_ASR_PROXY)
    return english


def normalize_transcript(text: str, lang: str) -> str:
    """Return native-language text for an ASR transcript (unchanged if already native)."""
    if not looks_like_transliterated_english(text, lang):
        return text
    english, _ = translate.to_english(_fix_leading_word(text), lang)
    native, _ = translate.from_english(english, lang)
    logger.debug(f"Code-mix normalised [{lang}]: {text!r} → {english!r} → {native!r}")
    return native or text


def transcribe_in_script(audio_bytes: bytes, lang: str, decode_mode: str | None = None) -> dict:
    """ASR, then return the transcript in the selected language's script."""
    if lang == "en":
        result = asr.transcribe(audio_bytes, lang=ENGLISH_ASR_PROXY, decode_mode=decode_mode)
        result["text"] = english_from_proxy_transcript(result["text"])
        return result
    result = asr.transcribe(audio_bytes, lang=lang, decode_mode=decode_mode)
    result["text"] = normalize_transcript(result["text"], lang)
    return result
