# Sticking corpus schema

The data contract for the learned sticking grammar (leg 3 of the fusion model).
Format: **JSONL**, one JSON object per *phrase* (a self-contained sticking line:
a rudiment cycle, an exercise row, a groove/fill bar). Estimated into a
context-conditioned back-off n-gram by `src/grammar.py`. No audio required.

## Phrase object

| field | type | notes |
|-------|------|-------|
| `source` | str | book + locus, e.g. `"stick_control_p6_ex25"`, `"wilcoxon_solo15"` |
| `provenance` | str | **`enumeration` \| `rudiment` \| `applied` \| `convention`** — see weighting note |
| `family` | str | `"single_paradiddle"`, `"flam_tap"`, `"linear_funk"`, … (free but consistent) |
| `lead` | str | `"R"` or `"L"` — which hand leads this written form (mirrors are separate phrases) |
| `subdivision` | str\|null | `"8th"`, `"16th"`, `"triplet"`, `"sextuplet"`, … (metric grid the `pos` indices ride) |
| `speed_tier` | str\|null | `"slow"` \| `"fast"` \| null — density/tempo class; null when the source is tempo-agnostic (most method books) |
| `cyclic` | bool | true if the phrase loops (last stroke transitions back to first); rudiment cycles are cyclic |
| `strokes` | list | ordered list of stroke objects, in *played* order (grace notes precede their primary) |

## Stroke object

| field | type | notes |
|-------|------|-------|
| `limb` | str | **`"L"` \| `"R"` \| `"F"`** (matches `types.LIMBS`; `F` = foot/kick) |
| `surface` | str | `kick/snare/tom/hihat/cymbal` (matches `types.SURFACES`); default `"snare"` for snare-method material |
| `accent` | bool | accented (`>`) stroke |
| `grace` | bool | grace note (the soft note of a flam/drag); a flam = 1 grace, a drag = 2 graces |
| `pos` | number\|null | metric position within the phrase on the `subdivision` grid (0-based index, or beat-fraction). null = unknown/uniform |

## Provenance weighting (the circularity guard)

The estimator multiplies each phrase's stroke counts by a provenance weight so
that **exhaustive permutation enumeration cannot wash out real human choices**:

- `enumeration` (Stick Control permutation pages) → low weight; serves as a
  smoothed back-off *background* of what is physically playable, not as evidence
  of what drummers *choose*.
- `rudiment` (canonical, evolved patterns) → full weight.
- `applied` (real grooves/fills with notated sticking) → full weight; the signal
  that lets the grammar transcend pure rudiment theory.
- `convention` (rules stated in prose, encoded as exemplar phrases) → full weight.

Default weights live in `src/grammar.py` (`PROVENANCE_WEIGHT`) so they are tunable
in one place and visible to eval.

### Provenance is material *type*, not source book (critical)

`enumeration` means **systematic permutation material** — a grid that runs an idea
through every rotation/combination — *regardless of which book it appears in*. A
permutation table in Garibaldi or Chaffee is `enumeration`, not `applied`. Only
**genuine in-context musical sticking choices** (a real groove/fill as actually
played) earn `applied`. Tagging on book identity instead of material type would
re-open the circularity trap: method books teach *vocabulary*, not *usage
frequency*, so raw permutation counts are not evidence of what drummers choose.

### Curation over volume (ingest discipline)

Do **not** transcribe permutation runs. From applied material, extract each
**distinct musical idea once** — a book printing 200 permutations of a cell should
contribute a handful of patterns, not 200. The grammar is a *weak validity +
local-transition* prior; pattern *selection* is done at inference by the audio
likelihood and metric/density context, not by corpus frequency. `grammar.fit()`
deduplicates identical phrases and reports the per-provenance mass balance so that
permutation-flooding cannot happen silently.
