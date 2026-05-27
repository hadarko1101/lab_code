from __future__ import annotations

import json


def print_result(result: dict) -> None:
    print(f"Source: {result['source']}")
    print(f"Sample rate: {result['sample_rate_hz']:.2f} Hz")
    print(f"Modulation: {result['modulation']}")
    print(f"Bits decoded: {result['bit_count']}")
    if "plots" in result:
        print(f"Original plot: {result['plots']['original']}")
        print(f"Envelope plot: {result['plots']['envelope']}")

    if result["protocol"] is None:
        print("Protocol: not detected")
        print(f"Reason: {result['details'].get('reason', 'unknown')}")
        return

    print(f"Protocol: {result['protocol']}")
    print(f"Card ID: {result['card_id']}")


def result_to_json(result: dict) -> str:
    return json.dumps(result, indent=2, sort_keys=True)
