from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path

from dotenv import load_dotenv

from beyond_click_sim.data.kuairand_translation import (
    TRANSLATION_FORMAT_VERSION,
    TRANSLATION_PROMPT_VERSION,
    atomic_write_csv,
    atomic_write_json,
    assemble_item_translations,
    category_batches,
    category_nodes,
    item_batches,
    load_eligible_items,
    sha256_file,
    total_usage,
    translate_batches,
    translation_audit,
)
from beyond_click_sim.llm_clients import make_llm_client


DEFAULT_MODEL = "gpt-5.5"
DEFAULT_CLIENT = "openai_vk_proxy"


def main() -> None:
    args = parse_args()
    items_path = args.items_parquet.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    items = load_eligible_items(items_path)
    if args.limit is not None:
        items = items.head(args.limit).copy()
    nodes = category_nodes(items)
    item_work = item_batches(items, batch_size=args.item_batch_size)
    category_work = category_batches(
        nodes,
        batch_size=args.category_batch_size,
    )
    preflight = {
        "event": "preflight",
        "model": args.model,
        "client": args.client,
        "items": len(items),
        "category_nodes": len(nodes),
        "item_batches": len(item_work),
        "category_batches": len(category_work),
        "caption_characters": int(items["caption"].str.len().sum()),
        "cover_text_characters": int(
            items["show_cover_text"].str.len().sum()
        ),
        "partial": args.limit is not None,
    }
    print(json.dumps(preflight, ensure_ascii=False, indent=2), flush=True)
    if args.preflight_only:
        return

    if args.env_file is not None:
        load_dotenv(args.env_file.expanduser().resolve(), override=False)
    client = make_llm_client(args.client)
    started_at = datetime.now(UTC)
    category_results = translate_batches(
        client=client,
        batches=category_work,
        checkpoint_dir=output_dir / "checkpoints",
        model=args.model,
        workers=args.workers,
        max_attempts=args.max_attempts,
        max_tokens=args.max_tokens,
    )
    item_results = translate_batches(
        client=client,
        batches=item_work,
        checkpoint_dir=output_dir / "checkpoints",
        model=args.model,
        workers=args.workers,
        max_attempts=args.max_attempts,
        max_tokens=args.max_tokens,
    )
    translated, glossary = assemble_item_translations(
        items=items,
        item_results=item_results,
        nodes=nodes,
        category_results=category_results,
    )
    audit = translation_audit(
        output=translated,
        category_glossary=glossary,
    )
    items_output = output_dir / "kuairand_items_en.csv"
    glossary_output = output_dir / "kuairand_category_glossary_en.csv"
    atomic_write_csv(items_output, translated)
    atomic_write_csv(glossary_output, glossary)

    all_results = [*category_results, *item_results]
    manifest = {
        "format_version": TRANSLATION_FORMAT_VERSION,
        "prompt_version": TRANSLATION_PROMPT_VERSION,
        "dataset": "kuairand-pure",
        "task": "official semantic supplement Chinese-to-English translation",
        "client": args.client,
        "model": args.model,
        "temperature": "provider_default",
        "source": {
            "items_parquet": str(items_path),
            "items_parquet_sha256": sha256_file(items_path),
        },
        "selection": {
            "requires_caption_or_cover_text": True,
            "requires_category_path": True,
            "items": len(items),
            "partial": args.limit is not None,
            "limit": args.limit,
        },
        "batching": {
            "item_batch_size": args.item_batch_size,
            "category_batch_size": args.category_batch_size,
            "workers": args.workers,
            "max_attempts": args.max_attempts,
            "max_completion_tokens_per_request": args.max_tokens,
            "item_batches": len(item_work),
            "category_batches": len(category_work),
        },
        "cache": {
            "checkpoint_root": "checkpoints",
            "new_batches": sum(not result.cached for result in all_results),
            "cached_batches": sum(result.cached for result in all_results),
        },
        "usage_including_cached_batches": total_usage(all_results),
        "audit": audit,
        "outputs": {
            "items": items_output.name,
            "items_sha256": sha256_file(items_output),
            "category_glossary": glossary_output.name,
            "category_glossary_sha256": sha256_file(glossary_output),
        },
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    atomic_write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Translate KuaiRand caption, cover text, and category metadata "
            "from Chinese to English through an OpenAI-compatible client."
        )
    )
    parser.add_argument("--items-parquet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--client", default=DEFAULT_CLIENT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--env-file",
        type=Path,
        help=(
            "Optional existing dotenv file used only to configure the client. "
            "The file is not copied into the output or recorded in the manifest."
        ),
    )
    parser.add_argument("--item-batch-size", type=int, default=20)
    parser.add_argument("--category-batch-size", type=int, default=40)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    for field in (
        "item_batch_size",
        "category_batch_size",
        "workers",
        "max_attempts",
        "max_tokens",
    ):
        if getattr(args, field) <= 0:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    return args


if __name__ == "__main__":
    main()
