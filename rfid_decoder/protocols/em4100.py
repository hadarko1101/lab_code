from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EM4100Result:
    card_id: str
    raw_bits: str
    nibbles: list[int]


def decode_em4100(bits: list[int]) -> EM4100Result | None:
    """Decode EM4100 64-bit frames.

    Frame shape:
    9 header ones, 10 rows of 4 data bits plus row parity, 4 column parity bits,
    and a stop zero.
    """
    frame = _find_frame(bits)
    if frame is None:
        return None

    rows = frame[9:59]
    column_parity = frame[59:63]
    stop = frame[63]
    if stop != 0:
        return None

    nibbles: list[int] = []
    columns = [0, 0, 0, 0]
    for row_index in range(10):
        chunk = rows[row_index * 5 : row_index * 5 + 5]
        data = chunk[:4]
        parity = chunk[4]
        if (sum(data) + parity) % 2 != 0:
            return None
        for index, bit in enumerate(data):
            columns[index] += bit
        nibbles.append(int("".join(str(bit) for bit in data), 2))

    for index, parity_bit in enumerate(column_parity):
        if (columns[index] + parity_bit) % 2 != 0:
            return None

    payload_nibbles = nibbles[2:]
    card_id = "".join(f"{nibble:X}" for nibble in payload_nibbles)
    return EM4100Result(card_id=card_id, raw_bits=_bits_to_string(frame), nibbles=nibbles)


def _find_frame(bits: list[int]) -> list[int] | None:
    for start in range(0, max(0, len(bits) - 63)):
        frame = bits[start : start + 64]
        if len(frame) == 64 and frame[:9] == [1] * 9:
            return frame
    return None


def _bits_to_string(bits: list[int]) -> str:
    return "".join(str(bit) for bit in bits)
