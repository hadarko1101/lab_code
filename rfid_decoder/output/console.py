from __future__ import annotations


def print_result(result: dict) -> None:
    print(f"Source: {result['source']}")
    print(f"Sample rate: {result['sample_rate_hz']:.2f} Hz")
    print(f"Modulation: {result['modulation']}")
    print(f"Bits decoded: {result['bit_count']}")

    if result["protocol"] is None:
        print("Protocol: not detected")
        print(f"Reason: {result['details'].get('reason', 'unknown')}")
        return

    print(f"Protocol: {result['protocol']}")
    print(f"Card ID: {result['card_id']}")
