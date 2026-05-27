from __future__ import annotations

from dataclasses import dataclass, field

from rfid_decoder.protocols.em4100 import decode_em4100
from rfid_decoder.protocols.wiegand35 import decode_wiegand35


@dataclass(frozen=True)
class ProtocolResult:
    protocol: str | None
    card_id: str | None
    details: dict = field(default_factory=dict)


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
            },
        )

    return ProtocolResult(
        protocol=None,
        card_id=None,
        details={"reason": "No supported protocol frame found"},
    )
