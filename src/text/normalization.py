"""
Convert between integer labels (1000-999999) and Russian spoken-word form.

Training labels are digit strings, but the audio contains Russian words,
so we normalize labels → words for CTC targets and denormalize predictions
back → digit strings for Kaggle submission.
"""

import re
from num2words import num2words


def num_to_words(n: int) -> str:
    """Convert integer to Russian spoken form, lower-cased and whitespace-normalised."""
    text = num2words(int(n), lang='ru')
    text = text.strip().lower()
    text = re.sub(r'\s+', ' ', text)
    return text


# ── Russian word→number tables ────────────────────────────────────────────────

_ONES = {
    'ноль': 0, 'нуль': 0,
    'один': 1, 'одна': 1,
    'два': 2, 'две': 2,
    'три': 3, 'четыре': 4,
    'пять': 5, 'шесть': 6, 'семь': 7, 'восемь': 8, 'девять': 9,
    'десять': 10,
    'одиннадцать': 11, 'двенадцать': 12, 'тринадцать': 13,
    'четырнадцать': 14, 'пятнадцать': 15, 'шестнадцать': 16,
    'семнадцать': 17, 'восемнадцать': 18, 'девятнадцать': 19,
}

_TENS = {
    'двадцать': 20, 'тридцать': 30, 'сорок': 40,
    'пятьдесят': 50, 'шестьдесят': 60, 'семьдесят': 70,
    'восемьдесят': 80, 'девяносто': 90,
}

_HUNDREDS = {
    'сто': 100, 'двести': 200, 'триста': 300, 'четыреста': 400,
    'пятьсот': 500, 'шестьсот': 600, 'семьсот': 700,
    'восемьсот': 800, 'девятьсот': 900,
}

_THOUSAND_WORDS = {'тысяча', 'тысячи', 'тысяч'}


def _parse_chunk(words: list[str]) -> int:
    """Parse a list of Russian words representing 0–999."""
    value = 0
    for w in words:
        if w in _HUNDREDS:
            value += _HUNDREDS[w]
        elif w in _TENS:
            value += _TENS[w]
        elif w in _ONES:
            value += _ONES[w]
    return value


def words_to_num(text: str) -> str:
    """
    Convert Russian spoken number words back to digit string.

    Returns the digit string, or the original text if parsing fails.
    """
    text = text.strip().lower()
    text = re.sub(r'\s+', ' ', text)
    words = text.split()

    # Locate first thousand marker
    thou_pos = next((i for i, w in enumerate(words) if w in _THOUSAND_WORDS), -1)

    if thou_pos == -1:
        value = _parse_chunk(words)
    else:
        thousands = max(_parse_chunk(words[:thou_pos]), 1) * 1000
        remainder = _parse_chunk(words[thou_pos + 1:])
        value = thousands + remainder

    return str(value) if value > 0 else text
