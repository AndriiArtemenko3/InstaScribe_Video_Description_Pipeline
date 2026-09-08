# ID Event Light Bench v0 — dataset collection guide

How to collect, annotate, validate, split and freeze the first real 48-item
dataset using the model-blind tooling in
`src/instadescribe_benchmarks/id_event_light_v0/dataset/`.

```
DO NOT RUN E001/E002 (OR ANY MODEL) ON CANDIDATE CLIPS BEFORE DATASET_FREEZE_V1.
```

Dataset construction is decided from human-observable event semantics only.
The tooling never scores a clip and never imports a solver or model runtime.

## Target

48 items: 16 positive / 16 negative / 16 hard_negative.
Split (assigned by the tool, atomically per source group):
DEV 18 (6/6/6) · TEST 30 (10/10/10).

## Workspace

Choose a workspace root **outside any git repository** (media and
licensing-sensitive provenance never enter git history):

```bash
DS="python -m instadescribe_benchmarks.id_event_light_v0.dataset.cli --root <root>"
```

Layout: `clips/` (derived benchmark clips), `sources/` (originals, optional),
`metadata/items.jsonl` + `metadata/exclusions.jsonl`, `manifest.dev.json` /
`manifest.test.json`, `freeze/`.

## Workflow

```
1.  record/acquire clip (rights-cleared only)
2.  register source + rights            $DS add ...
3.  hash original                       (add --original <file> hashes it)
4.  derive benchmark clip if needed     (trim/convert; log via --transform)
5.  hash derived clip                   (add hashes it automatically)
6.  annotate                            $DS annotate ...
7.  review (explicit second pass)       $DS review ...
8.  repeat until 48                     $DS status
9.  assign split                        $DS assign-split
10. validate                            $DS validate
11. build manifests                     $DS build-manifests
12. freeze                              $DS freeze --version DATASET_FREEZE_V1
```

`$DS verify-freeze <root>/freeze/DATASET_FREEZE_V1.json` later proves nothing
drifted. Changing any clip, query, label, difficulty, boundary, source group
or split after freeze creates a NEW dataset version.

## Registering an item

```bash
$DS add --item-id pos-001 --source-group session-2026-09-10-a \
  --source-type self_recorded --clip <root>/clips/pos-001.mp4 \
  --original <root>/sources/IMG_1234.MOV \
  --source-reference "own phone recording, parking lot session A" \
  --transform "ffmpeg trim 00:12-00:31, container copy"
```

- `--source-group`: the leakage unit — same original video, same continuous
  recording/session, or near-duplicate family share ONE group id. Two
  independent recordings at the same location are DIFFERENT groups.
- Externally sourced media additionally requires `--licence-id` and
  `--licence-evidence` before freeze. The tool checks that evidence exists;
  it does not judge legal validity.
- Allowed derivations: temporal trim, container/codec conversion, resize.
  Never: speed changes, inserted/removed content, overlays, augmentation.

## Annotating

```bash
$DS annotate --item-id pos-001 --query "a person enters a vehicle" \
  --difficulty positive --category directional_transition \
  --start 12.4 --end 15.1 --annotator <name>
```

**Positive boundaries** (0.1 s precision; the tool surfaces any normalization):

- `start`: the first moment the queried action is visibly underway — not the
  preparatory approach.
- `end`: the first moment the resulting post-state is stably established.
- ±0.5 s is acknowledged as inherent ambiguity; set boundaries once, never
  adjust them after any model run.
- Prefer single-occurrence clips; otherwise annotate the FIRST complete
  occurrence and note the rest.

**Queries** are human-authored, concise declarative event phrases
("a person enters a vehicle"). No model-assisted rewording, ever.

**Event categories**: static_state, simple_motion, interaction, transition,
directional_transition, state_change.

**Hard negatives** (`--difficulty hard_negative --hard-negative-type <t>`):
visually/semantically close while the event does not occur —
precondition_without_action, postcondition_without_transition,
related_object_interaction, wrong_direction, partial_action, similar_motion,
same_scene_wrong_event, temporal_near_miss.

## Excluding

```bash
$DS exclude --item-id cand-017 --reason ambiguous_event --notes "..."
```

The ledger is append-only; reconsidering writes a new event
(`--decision reconsidered`), never erasing history. `--solver-output-seen`
must remain unset for every decision made before the first model run.

## Freeze

`freeze` enforces everything at once: 48 items, 16/16/16, 6/6/6 + 10/10/10,
all reviewed and assigned, rights complete, media present with matching
digests, no source-group leakage, valid annotations, manifests passing the
strict benchmark loader, unique ids and clip hashes. On success it writes
`freeze/DATASET_FREEZE_V1.json` (+ its sha256) recording every item id and
clip digest, both manifest digests, the metadata digest, source groups,
counts and the tool version.

After `DATASET_FREEZE_V1` exists, model development resumes with the
pre-registered E001↔E002 evaluation protocol.
