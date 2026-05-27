from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Wiegand35Result:
    facility_code: int
    card_number: int
    card_id: str
    raw_bits: str


def decode_wiegand35(bits: list[int]) -> Wiegand35Result | None:
    """Decode a common 35-bit Corporate 1000-style Wiegand layout.

    Layout used:
    bit 0: even parity over bits 1..17
    bits 1..12: facility code
    bits 13..33: card number
    bit 34: odd parity over bits 18..33
    """
    if len(bits) < 35:
        return None

    for start in range(0, len(bits) - 34):
        frame = bits[start : start + 35]
        if not _valid_parity(frame):
            continue
        facility = _to_int(frame[1:13])
        card_number = _to_int(frame[13:34])
        return Wiegand35Result(
            facility_code=facility,
            card_number=card_number,
            card_id=f"{facility}:{card_number}",
            raw_bits="".join(str(bit) for bit in frame),
        )
    return None


def _valid_parity(frame: list[int]) -> bool:
    even_region = frame[1:18]
    odd_region = frame[18:34]
    even_ok = (sum(even_region) + frame[0]) % 2 == 0
    odd_ok = (sum(odd_region) + frame[34]) % 2 == 1
    return even_ok and odd_ok


def _to_int(bits: list[int]) -> int:
    return int("".join(str(bit) for bit in bits), 2)
