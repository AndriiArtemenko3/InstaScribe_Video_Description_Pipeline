# Benchmark decisions — ID Event Light Bench v0

Concise log of the design decisions behind `packages/benchmarks`. Scope: v0
only; future tracks record their own decisions when they exist.

## 1. Event queries are natural-language strings

The benchmark tests the product's real interface — "did *this described thing*
happen?" — not membership in a fixed label set. A closed taxonomy would test a
classifier, not language grounding, and would freeze v0 around one event class.

## 2. Event detection and temporal localization are scored separately

A system can be right about *whether* and wrong about *when*. One combined
score would hide which skill failed. Classification is driven by
`event_detected` alone; temporal IoU and start/end errors are computed only
over true positives, with mean IoU averaged over all true positives so that
omitting timestamps never improves the headline number.

## 3. Hard negatives are explicitly represented

Plain negatives measure little — most clips contain no specified event. Hard
negatives (visually and semantically adjacent non-events: the door opens but
nobody enters) are where detection systems actually fail, so they form their
own bucket with its own false-positive rate.

## 4. GEO and retrieval are not part of v0

Geolocation, retrieval, entity recognition, tracking and change detection are
independent capabilities that deserve independently scored tracks (ID-GEO,
ID-RETRIEVAL, ...). Folding them into v0 would produce one unreadable
composite score and a harness too large to audit. v0 stays one skill.

## 5. The evaluator is model-agnostic

The evaluator scores a predictions JSON file and never runs a model. A
hand-written heuristic, a CNN, a tracker, a VLM and the full InstaDescribe
pipeline are scored identically, which keeps the harness dependency-free and
makes every future baseline comparison fair by construction.

## 6. A new snake_case schema family

The manifest uses snake_case keys and its own `schema_version`
(`id-event-light-bench-manifest/0`), deliberately diverging from the frozen
camelCase `instascribe-eval-manifest/1` family
(`tests/fixtures/evaluation/manifest.v1.json`). The benchmark spec's field
names (`start_time`, `event_detected`) are the contract, and an independent
family avoids implying compatibility with the frozen G8 evaluation manifest.
