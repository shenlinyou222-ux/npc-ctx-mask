# Dataset

We ship **annotations and pointers**, not the raw wiki text.

| File | Content |
|---|---|
| zzz_dataset_v0/ | Zenless Zone Zero 2.0–3.2 version-gated QA (built locally) |
| NOTICE.md | per-page attribution: title, revision id, URL, license |

## Source & license

Text is fetched at build time from **Zenless Zone Zero Wiki (Fandom)**, licensed **CC BY-SA**.
Redistribution of the derived annotations requires attribution and share-alike.
See NOTICE.md for the per-page list and scripts/zzz_dataset.py for the builder.

## Why version-gated

The character's knowledge is bounded by story progress: a character at 2.0 must not know
2.4 content. This gives a **natural, non-synthetic visibility label** —
no hand-made tenant partitioning.

Closed-book baseline on this set is **0/60** for the model we tested, i.e. the questions
are not answerable from parametric memory, so a correct answer must come from the context.
