from __future__ import annotations

from pprint import pprint

from tsdiag import diagnose
from tsdiag.result_contract import result_summary

from domain_scenarios import build_scenarios


def main() -> None:
    scenarios = build_scenarios()

    print("Time-Series Diagnostic Agent — standardized six-domain demo")
    print("=" * 72)
    for domain, cases in scenarios.items():
        print(f"\n[{domain}]")
        for case_name in ("normal", "fault"):
            result = diagnose(domain, **cases[case_name])
            compact = result_summary(result)
            print(f"  {case_name}:")
            pprint(compact, sort_dicts=False, width=100)


if __name__ == "__main__":
    main()
