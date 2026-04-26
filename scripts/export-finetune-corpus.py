#!/usr/bin/env python3
"""
export-finetune-corpus.py — Exportiert das gesammelte Bot-Memory-Material
in OpenAI-Fine-Tuning-Format (JSONL mit messages-Arrays).

Quellen:
- 06_Meta/bot-memory/conversation-history.jsonl  (alle Turns)
- 06_Meta/bot-memory/corrections.jsonl           (User-Korrekturen — am wertvollsten)
- 06_Meta/bot-memory/preferences.md, facts.md    (für System-Prompt-Konstruktion)

Output: training-corpus.jsonl mit Records wie:
  {"messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]}

Verwendung:
  python3 scripts/export-finetune-corpus.py \\
      --vault /opt/vault/KI_WIKI_Vault \\
      --user-id 5606448807 \\
      --output ./corpus.jsonl \\
      --include-corrections \\
      --include-conversations
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterator


def load_facts(facts_file: Path) -> str:
    if not facts_file.exists():
        return ""
    content = facts_file.read_text(encoding="utf-8").strip()
    # Strip frontmatter + first heading
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            content = parts[2].strip()
    lines = content.splitlines()
    while lines and (lines[0].startswith("#") or lines[0].strip() == ""):
        lines.pop(0)
    return "\n".join(lines).strip()


def build_system_prompt(facts: str, prefs: str) -> str:
    parts = ["Du bist Julius' persönlicher Vault-Assistent über Telegram. Antworte auf Deutsch, kurz und direkt."]
    if prefs:
        parts.append(f"\n# PRÄFERENZEN\n{prefs}")
    if facts:
        parts.append(f"\n# FAKTEN\n{facts}")
    return "\n".join(parts)


def iter_conversation_pairs(history_file: Path, user_id: int) -> Iterator[dict]:
    """Liefert (user_msg, assistant_msg) Paare aus der History."""
    if not history_file.exists():
        return
    user_msgs = []
    last_user = None
    with history_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("user_id") != user_id:
                continue
            msg = rec.get("msg", {})
            role = msg.get("role")
            content = msg.get("content", "")
            if isinstance(content, list):
                # tool_use_use blocks etc. → reduce to text
                content = " ".join(
                    str(c.get("text", "")) for c in content
                    if isinstance(c, dict) and c.get("type") == "text"
                )
            content = str(content).strip()
            if not content:
                continue
            if role == "user" and not content.startswith("[Upload-Event]"):
                last_user = content
            elif role == "assistant" and last_user:
                yield {"user": last_user, "assistant": content}
                last_user = None


def iter_corrections(corrections_file: Path) -> Iterator[dict]:
    if not corrections_file.exists():
        return
    with corrections_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                yield rec
            except Exception:
                continue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vault", type=Path, required=True, help="Pfad zum Vault-Root")
    ap.add_argument("--user-id", type=int, required=True, help="Telegram-User-ID (zum Filter)")
    ap.add_argument("--output", type=Path, default=Path("training-corpus.jsonl"))
    ap.add_argument("--include-corrections", action="store_true", help="Korrekturen als Trainings-Pairs")
    ap.add_argument("--include-conversations", action="store_true", help="Conversation-Pairs aus history")
    ap.add_argument("--max-conversations", type=int, default=500, help="Max conversation pairs")
    args = ap.parse_args()

    memory_dir = args.vault / "06_Meta" / "bot-memory"
    facts = load_facts(memory_dir / "facts.md")
    prefs = load_facts(memory_dir / "preferences.md")  # gleiche Stripping-Logik
    system_prompt = build_system_prompt(facts, prefs)

    history_file = memory_dir / "conversation-history.jsonl"
    corrections_file = memory_dir / "corrections.jsonl"

    records = []

    if args.include_corrections:
        for c in iter_corrections(corrections_file):
            user_msg = c.get("kontext") or c.get("was_falsch", "(kein Kontext)")
            assistant_msg = c.get("was_richtig", "")
            if not assistant_msg:
                continue
            records.append({
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": assistant_msg},
                ]
            })

    if args.include_conversations:
        pairs = list(iter_conversation_pairs(history_file, args.user_id))
        pairs = pairs[-args.max_conversations:]
        for p in pairs:
            records.append({
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": p["user"]},
                    {"role": "assistant", "content": p["assistant"]},
                ]
            })

    if not records:
        print("⚠️  Keine Records gefunden. Setze --include-corrections und/oder --include-conversations.", file=sys.stderr)
        sys.exit(1)

    with args.output.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"✓ {len(records)} Trainings-Records exportiert nach {args.output}")
    print(f"  System-Prompt-Länge: {len(system_prompt)} Zeichen")
    print(f"  Korrekturen: {sum(1 for r in records if 'kontext' in str(r))}")
    print()
    print("Nächste Schritte für Fine-Tuning:")
    print("  • OpenAI: https://platform.openai.com/finetune (Modell + corpus.jsonl hochladen)")
    print("  • Anthropic: https://docs.anthropic.com/en/api/fine-tuning")
    print("  • Mistral / Together AI / Replicate: jeweils API für Custom-Models")


if __name__ == "__main__":
    main()
