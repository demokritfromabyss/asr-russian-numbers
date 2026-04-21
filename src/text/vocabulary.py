"""
Character vocabulary for the Russian spoken-numbers domain.

All characters that appear in num2words(n, lang='ru') for n in [1000, 999999].
CTC blank is always index 0.
"""


# Every Russian character that appears in number words for this range
_CHARS = sorted([
    ' ',                                          # word separator
    'а', 'в', 'д', 'е', 'и', 'к', 'м', 'н',
    'о', 'п', 'р', 'с', 'т', 'ц', 'ч', 'ш',
    'ы', 'ь', 'я',
])

BLANK_ID = 0
VOCAB = ['<blank>'] + _CHARS          # len == 21
CHAR2ID = {c: i for i, c in enumerate(VOCAB)}
ID2CHAR = {i: c for i, c in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)


class Vocabulary:
    blank_id: int = BLANK_ID
    size: int = VOCAB_SIZE

    def encode(self, text: str) -> list[int]:
        """Map a string to a list of token ids, silently dropping unknown chars."""
        return [CHAR2ID[c] for c in text.lower() if c in CHAR2ID]

    def decode(self, ids: list[int]) -> str:
        """Map a list of token ids back to a string."""
        return ''.join(ID2CHAR.get(i, '') for i in ids if i != BLANK_ID)
