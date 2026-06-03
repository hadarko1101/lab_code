from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ProtocolResult:
    protocol: str | None
    card_id: str | None
    details: dict = field(default_factory=dict)


@dataclass(frozen=True)
class EM4100Result:
    card_id: str
    raw_bits: str
    nibbles: list[int]


@dataclass(frozen=True)
class Wiegand35Result:
    facility_code: int
    card_number: int
    card_id: str
    raw_bits: str
    repeat_count: int = 1


def autodetect_protocol(bits: list[int]) -> ProtocolResult:
    em = decode_em4100(bits)
    if em is not None:
        return ProtocolResult(
            protocol="EM4100",
            card_id=em.card_id,
            details={"raw_bits": em.raw_bits, "nibbles": em.nibbles},
        )

    wiegand = decode_wiegand35(bits)
    if wiegand is not None:
        return ProtocolResult(
            protocol="Wiegand35",
            card_id=wiegand.card_id,
            details={
                "facility_code": wiegand.facility_code,
                "card_number": wiegand.card_number,
                "raw_bits": wiegand.raw_bits,
                "repeat_count": wiegand.repeat_count,
            },
        )

    return ProtocolResult(
        protocol=None,
        card_id=None,
        details={"reason": "No supported protocol frame found"},
    )


def decode_em4100(bits: list[int]) -> EM4100Result | None:
    """Decode EM4100 64-bit frames."""
    frame = _find_em4100_frame(bits)
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


def decode_wiegand35(bits: list[int]) -> Wiegand35Result | None:
    """Decode a common 35-bit Corporate 1000-style Wiegand layout."""
    if len(bits) < 35:
        return None

    candidates: dict[str, int] = {}
    for start in range(0, len(bits) - 34):
        frame = bits[start : start + 35]
        if not _valid_wiegand35_parity(frame):
            continue
        raw_bits = _bits_to_string(frame)
        candidates[raw_bits] = candidates.get(raw_bits, 0) + 1

    if not candidates:
        return None

    raw_bits, repeat_count = max(candidates.items(), key=lambda item: item[1])
    frame = [int(bit) for bit in raw_bits]
    facility = _bits_to_int(frame[1:13])
    card_number = _bits_to_int(frame[13:34])
    return Wiegand35Result(
        facility_code=facility,
        card_number=card_number,
        card_id=f"{facility}:{card_number}",
        raw_bits=raw_bits,
        repeat_count=repeat_count,
    )


def decode_repeated_wiegand35(bits: list[int], *, min_repeats: int = 2) -> Wiegand35Result | None:
    result = decode_wiegand35(bits)
    if result is None or result.repeat_count < min_repeats:
        return None
    return result


def _find_em4100_frame(bits: list[int]) -> list[int] | None:
    for start in range(0, max(0, len(bits) - 63)):
        frame = bits[start : start + 64]
        if len(frame) == 64 and frame[:9] == [1] * 9:
            return frame
    return None


def _valid_wiegand35_parity(frame: list[int]) -> bool:
    even_region = frame[1:18]
    odd_region = frame[18:34]
    even_ok = (sum(even_region) + frame[0]) % 2 == 0
    odd_ok = (sum(odd_region) + frame[34]) % 2 == 1
    return even_ok and odd_ok


def _bits_to_int(bits: list[int]) -> int:
    return int(_bits_to_string(bits), 2)


def _bits_to_string(bits: list[int]) -> str:
    return "".join(str(bit) for bit in bits)
