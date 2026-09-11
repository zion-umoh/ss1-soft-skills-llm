"""Create the final contextual comparison with the published RecruitView CRMF paper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PAPER_BIG_FIVE = {
    "openness": 0.6384,
    "conscientiousness": 0.5572,
    "extraversion": 0.5681,
    "agreeableness": 0.5927,
    "neuroticism": 0.2603,
}
PAPER_SPEAKING_SKILLS = 0.5947


def build(model_report: Path, output_json: Path, output_markdown: Path) -> dict[str, object]:
    report = json.loads(model_report.read_text(encoding="utf-8"))
    selected = report["protocol"]["primary_variant"]
    ours = report["variants"][selected]
    ours_big_five = {key: value["spearman"] for key, value in ours["big_five"]["test_metrics"]["per_target"].items()}
    ours_speaking = ours["speaking_skills"]["test_metrics"]["per_target"]["speaking_skills"]["spearman"]
    payload = {
        "status": "completed",
        "reference": {
            "paper": "Gupta et al. (2025), RecruitView, arXiv:2512.00450",
            "configuration": "CRMF (VideoMAE + Wav2Vec2)",
            "metric": "Spearman rho",
            "big_five": PAPER_BIG_FIVE,
            "speaking_skills": PAPER_SPEAKING_SKILLS,
        },
        "ours": {
            "model_report": str(model_report),
            "selected_variant": selected,
            "metric": "participant-disjoint test Spearman rho",
            "big_five": ours_big_five,
            "speaking_skills": ours_speaking,
        },
        "difference_ours_minus_paper": {
            "big_five": {key: ours_big_five[key] - PAPER_BIG_FIVE[key] for key in PAPER_BIG_FIVE},
            "speaking_skills": ours_speaking - PAPER_SPEAKING_SKILLS,
        },
        "interpretation": "Contextual comparison only: the paper uses its own multimodal CRMF model, split and full target set, whereas this study uses a participant-disjoint local split, engineered audio columns, an LLM-assisted structured text representation and a selected Ridge variant after excluding empty transcripts.",
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Final RecruitView comparison with published CRMF",
        "",
        "This is a contextual benchmark, not an exact reproduction: the paper uses multimodal CRMF with VideoMAE + Wav2Vec2 and its own split, while this study uses the fixed participant-disjoint split, engineered audio columns and an LLM-assisted structured text representation.",
        "",
        "## Big Five (Spearman ρ)",
        "",
        "| Target | Our selected variant | Published CRMF | Difference |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key in PAPER_BIG_FIVE:
        lines.append(f"| {key} | {ours_big_five[key]:.4f} | {PAPER_BIG_FIVE[key]:.4f} | {ours_big_five[key] - PAPER_BIG_FIVE[key]:+.4f} |")
    lines.extend(
        [
            "",
            "## Direct `speaking_skills` benchmark",
            "",
            "| Our selected variant | Published CRMF | Difference |",
            "| ---: | ---: | ---: |",
            f"| {ours_speaking:.4f} | {PAPER_SPEAKING_SKILLS:.4f} | {ours_speaking - PAPER_SPEAKING_SKILLS:+.4f} |",
            "",
            "The published values are not a pass/fail threshold. They show the context in which this smaller, engineered-feature experiment should be interpreted.",
            "",
        ]
    )
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.write_text("\n".join(lines), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-report", type=Path, default=Path("reports/benchmark/recruitview-feature-fusion.json"))
    parser.add_argument("--output-json", type=Path, default=Path("reports/benchmark/recruitview-paper-comparison.json"))
    parser.add_argument("--output-markdown", type=Path, default=Path("reports/benchmark/recruitview-paper-comparison.md"))
    args = parser.parse_args()
    result = build(args.model_report, args.output_json, args.output_markdown)
    print(json.dumps({"status": result["status"], "selected_variant": result["ours"]["selected_variant"]}, indent=2))


if __name__ == "__main__":
    main()
