# KuaiRand semantic translation with GPT-5.5

## Result

The released KuaiRand-Pure Chinese semantic metadata was translated through
the `openai_vk_proxy` client using model `gpt-5.5`.

- Eligible items: 7,506.
- Non-empty captions translated: 7,486 / 7,486.
- Non-empty cover texts translated: 942 / 942.
- Category paths translated: 7,506 / 7,506.
- Unique category nodes translated once and reused: 512.
- Item batches: 376 at 20 items per request.
- Category batches: 13 at 40 nodes per request.
- Failed or missing batches: 0.
- Prompt tokens: 459,430.
- Completion tokens: 692,403.
- Total provider-reported tokens: 1,151,833.

The first run created all 389 content-addressed checkpoints. An exact replay
loaded 389 / 389 checkpoints, made zero new translation batches, and reproduced
byte-identical output hashes.

## Output

Local generated artifacts are intentionally outside Git:

```text
data/processed/kuairand/semantic_translation_gpt55/
├── checkpoints/
├── kuairand_category_glossary_en.csv
├── kuairand_items_en.csv
└── manifest.json
```

Hashes:

```text
kuairand_items_en.csv
4fca7b9153c73e0b5206357b95e0ebee2759a1647f2d366a7fcf8078f492a1eb

kuairand_category_glossary_en.csv
f1cf0e32d6d148dac958c6b924e12b599fa0a303c9a5664b7c50a731ebef152b
```

## Translation contract

The prompt asks for faithful, concise English while preserving proper nouns,
media/game titles, numbers, emoji, hashtags, and `@mentions`. Caption and cover
text remain separate; one is not used as a fallback for the other. Category
nodes are translated through a single glossary so the same Chinese category
has the same English value in every path.

The response parser requires:

- the exact requested IDs;
- the exact JSON fields;
- no duplicate keys or records;
- source/translation empty-field parity.

Every valid batch is written atomically to a checkpoint. The checkpoint key
hashes the source records, prompt, model, request settings, and format version.

## Quality audit

All eligible rows and non-empty fields survived translation. All 512 English
category labels contain no CJK characters.

Chinese characters remain in 1,632 English captions and eight English cover
texts. Manual inspection shows that this is dominated by deliberately
preserved Chinese `@mentions`, hashtags, song/title strings, and proper names.
For example, ordinary prose around the handles is translated while
`@快手热点(O40300129)` remains unchanged. This is preferable to corrupting
identifiers that may carry semantic or entity information.

After mechanically removing simple `@mentions` and hashtags for audit only, 33
captions and three cover texts still contain CJK. Inspection found mostly
multi-token account names, artist/title strings, game-specific names, and a few
isolated mixed-language terms such as `Shooting篇`; this is a review queue, not
evidence that 1,632 descriptions were left untranslated.

Deterministic source/translation samples were inspected for captions, cover
texts, and category trees. No missing translation or systematic block of
untranslated prose was found. A later sensitivity run may remove handles and
hashtags from both languages if their script creates a model-specific
advantage, but the primary metadata should retain released content.

## Reproduction

```bash
PYTHONPATH=src .venv/bin/python scripts/translate_kuairand_semantics.py \
  --items-parquet data/canonical/kuairand-pure/v1/items.parquet \
  --output-dir data/processed/kuairand/semantic_translation_gpt55 \
  --env-file /path/to/existing/repo/.env \
  --client openai_vk_proxy \
  --model gpt-5.5
```

The environment file is read only to construct the client. Its path and
contents are not copied to the output manifest.

The final manifest reflects the verification replay (`new_batches=0`,
`cached_batches=389`). Token usage remains available because it is retained in
the checkpoints. The initial paid run created 389 new batches.
