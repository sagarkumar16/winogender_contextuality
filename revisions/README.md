# EMNLP revision experiments

Everything here is **additive**. No file outside `revisions/` was modified except
`requirements.txt` (two dependencies appended, see below). The published analysis code is
imported, never edited.

```
revisions/
  cbd.py                    joint-measurement contextuality math (the new analysis math)
  common.py                 seeding, provenance, IO, count/probability helpers
  primes.py                 Exp 3: build referent-only null primes            [CLI]
  collect_joint.py          Exp 1: collect joint two-pronoun measurements     [CLI, GPU]
  collect_unquantized.py    Exp 2: re-run the pipeline unquantized            [CLI, GPU]
  collect_referent_null.py  Exp 3: collect referent-only null measurements    [CLI, GPU]
  joint_measurement.py      Exp 1: steering vs joint ΔC                       [CLI]
  quantization_compare.py   Exp 2: quantized vs unquantized + sanity check    [CLI]
  referent_null.py          Exp 3: three-condition KL and ΔC                  [CLI]
  per_item_table.py         Exp 4: per-item plain probabilities (CSV + LaTeX) [CLI]
  fixtures.py               synthetic NDJSON for tests and GPU-free smoke runs[CLI]
  collect_common.py         shared collection machinery
  configs/                  model sets (models_fast / models_paper / models_unquantized)
  slurm/                    cluster job scripts
  tests/                    pytest suite (74 tests)
  outputs/                  all results land here; nothing existing is overwritten
```

Run the tests with `pytest revisions/tests` (they need no GPU and no measurement data).

## Running everything on the cluster

```bash
# 1. submit the GPU collection jobs (login node; NOT itself a SLURM job)
bash revisions/slurm/submit_all.sh --dry-run     # see the plan first
bash revisions/slurm/submit_all.sh

#    ...or start small, which is what I'd do:
EXPERIMENTS=joint MODELS="meta-llama/Llama-3.2-1B-Instruct" bash revisions/slurm/submit_all.sh

# 2. once the jobs land, run all four analyses (CPU only, no SLURM)
bash revisions/slurm/run_analyses.sh
```

`run_analyses.sh` skips any experiment whose inputs are not there yet and says so, so it is
safe to re-run as jobs finish. Experiment 4 needs no new collection — it reads the published
`one_pronoun_measurements_*.ndjson`.

| script | what it does |
|---|---|
| `slurm/submit_all.sh` | sbatch's one job per (experiment × model); `--dry-run` prints the plan. Raises `--mem` for the unquantized arm (bf16 weights are much larger than 4-bit) |
| `slurm/collect_joint.sh` | Exp 1 — joint two-pronoun measurements |
| `slurm/collect_unquantized.sh` | Exp 2 — `CONDITION=primed\|null\|both` |
| `slurm/collect_referent_null.sh` | Exp 3 — builds the primes if missing, then collects |
| `slurm/run_analyses.sh` | all four analyses, CPU only |

Each collection script takes its knobs from the environment (`MODEL`, `TEMP`, `N_RUNS`,
`SEED`, `BATCH_SIZE`, `DATA_DIR`), defaulting to the paper's values (temperature 0.5,
50 runs, batches of 10). They **derive the row count from the input file** rather than
hardcoding it — see the warning below.

> **Check this before launching the null runs.** The existing `collect_null.sh` hardcodes
> `N_ROWS=181`, but its input `all_sentences_wp.csv` is indexed **per sentence**, not per pair,
> and the paper notebook reads null data with `max_index=360`. If that file has ~362 rows, the
> published null runs covered only its first half. I could not check — the file is not in the
> repo. My scripts read the row count off the file, so they are not exposed to this, but the
> existing null results may be.

---

## Read this first: three defects found in the existing analysis code

These were found while building the comparison. **Nothing was changed** — the constraint was
no edits to existing code — but they affect published numbers, so they need a decision before
the response letter goes out.

### 1. The steering ΔC uses the wrong marginal (affects reported ΔC)

`contextuality.calculate_sentence_dc_fraction` builds the forward correlation as

```python
cbd_correlation(V1, V2, V1W2)     # V1 = P(prime=f | fwd),  V2 = P(prime=f | rev)
```

CbD calls for the two marginals **of that context** — the prime marginal and the *generation*
marginal, `(V1, W2)`. `V2` is the prime marginal of the *other* context. The reverse term is
built correctly, and the inconsistent-connectedness term is correct.

The two expressions coincide only when `V2 == W2`. Primes are balanced by design so
`V2 = 0.5`; therefore the published ΔC is correct exactly on items whose forward generation
marginal happens to be 0.5, and wrong elsewhere. The error can flip the verdict:
`revisions/tests/test_joint_vs_steering.py::test_diverges_from_published_estimator_when_generation_marginal_is_skewed`
constructs an item where the correct ΔC is 0.0 (not contextual) and the published ΔC is +0.4
(contextual).

`joint_measurement.py` therefore reports **both** `delta_c_steering_published` (the published
estimator, imported unmodified) and `delta_c_steering` (the same data, same estimator as the
joint column). Compare them before deciding what to put in the paper.

### 2. `mode='internal'` ΔC has an identically-zero joint probability

In the same function, the internal branch computes the joint count as

```python
forward_trials = zip(data_dict['forward']['fixed_pnoun'],   # strings: 'he' / 'she'
                     data_dict['forward']['free_pnoun'])    # LISTS of logits
count_c1 = sum(1 for x, y in forward_trials if x == target_f[0] and y == target_f[1])
```

`y` is a logit vector and `target_f[1]` is a pronoun string, so `y == target_f[1]` is never
true and `V1W2 == 0` for every item. Any ΔC computed with `mode='internal'` is therefore not
measuring a joint distribution at all. The generation-mode numbers are unaffected.

### 3. The null runs store logits over the characters of "she"

`collect_sequential.generate_one_null_context` does:

```python
pronouns = ast.literal_eval(df.differences[idx])   # a LIST: ['he', 'she']
pronoun_idxs = mp.get_token_ids(options=pronouns[1])   # pronouns[1] is the STRING 'she'
```

`get_token_ids` iterates its argument, so it tokenises `'s'`, `'h'`, `'e'` and the `logits`
field of every existing `null_measurements_*.ndjson` is a 3-vector over those characters, not
a 2-vector over `['he','she']`. (`generate_one_pronoun` is fine: there `pronouns` is a *dict*,
so `pronouns[1]` is the option list.)

The published null analysis uses generation counts, so **its numbers stand**. But no
logit-based null quantity can be computed from the existing files. Experiment 3 is therefore
reported on generation counts, and our new referent-null collector stores proper 2-vectors.

### Also worth knowing

* `contextuality.cbd_s1_4cycle` includes the term `|w + x - y - z|`, which has an **even**
  number of minus signs. The CbD `s1` maximises only over **odd**-sign patterns; the correct
  fourth term is `|w + x - y + z|`. `revisions/cbd.py:s1_cyclic` enumerates the odd patterns
  honestly. This affects `calculate_pronouns_nc_fraction` / `measure_contextuality.py`, which
  the paper's figures do not appear to use.
* `contextuality.pronoun_context_array` orders its four contexts `(0,0), (0,1), (1,0), (1,1)`.
  A rank-4 cyclic system requires *consecutive contexts to share a content variable*, and
  `(0,1)` and `(1,0)` share none. `joint_measurement.joint_rank4` uses the cycle order
  `(0,0), (1,0), (1,1), (0,1)`.
* `collect_sequential.generate_two_pronouns` constructs `Context` without `case_1`/`case_2`,
  which are required fields, so it raises `TypeError` before writing a record. That is why
  Experiment 1 needed a new joint collector rather than a call into the existing one.

---

## Setup

```bash
pip install -r requirements.txt      # scipy + statsmodels appended; statsmodels was missing
                                     # and `import winogender_contextuality.utils` needs it
```

Inputs are read from `INTERIM_DATA_DIR` (see `winogender_contextuality/config.py`). Every CLI
takes `--data-dir` to point elsewhere, or set `$WC_INTERIM_DIR`. Results are written under
`revisions/outputs/` (override with `--out`); no existing results directory is ever touched.

Every results file gets a `<name>.meta.json` sidecar with the seed, git commit, library
versions, model name/revision, quantization setting and the exact command line.

`--max-index` defaults to 180, which drops the `['she','potato']` sentinel row that
`dataset.py` appends — matching the paper's analyses.

### Smoke-test without a GPU

```bash
python -m revisions.fixtures --out /tmp/fix
python -m revisions.joint_measurement --steering /tmp/fix/steering.ndjson \
    --joint /tmp/fix/joint.ndjson --models synthetic --max-index 4 --out /tmp/out
```

---

## Experiment 1 — Joint-measurement contextuality (Reviewer 1, "Steering")

The published runs are **steering** measurements: the priming pronoun is fixed, so only the
free pronoun is measured. This experiment measures **both** pronouns in a single pass
(`BLANK1`/`BLANK2`), which is the formulation the QQ equality and the social-science
contextuality literature use, and scores both with the same estimator.

Both formulations are cyclic systems, so one implementation serves both
(`revisions/cbd.py`). For a rank-2 system,

```
ΔC = |<R_A R_B>_fwd - <R_A R_B>_rev| - |<R_A>_fwd - <R_A>_rev| - |<R_B>_rev - <R_B>_fwd|
```

contextual iff ΔC > 0. **Connection to the reviewer's point:** for two binary questions asked
in both orders the QQ statistic is `q = P_fwd(agree) - P_rev(agree)`, and since
`<XY> = 2·P(agree) - 1`, the identity

```
ΔC(rank 2) = 2·|q| - Δ0
```

holds exactly (tested in `test_cbd.py::test_delta_c_equals_two_abs_q_minus_disturbance`). The
joint ΔC *is* the QQ order effect, corrected for inconsistent connectedness. `qq` is reported
per item so the QQ equality can be tested directly.

```bash
# collect (GPU) -- defaults to the small, fast model set
MODEL=meta-llama/Llama-3.2-1B-Instruct sbatch revisions/slurm/collect_joint.sh

# analyse
python -m revisions.joint_measurement \
    --steering one_pronoun_measurements_Llama-3.2-1B-Instruct_0.5_wp.ndjson \
    --joint    joint_measurements_Llama-3.2-1B-Instruct_0.5_wp.ndjson \
    --models   llama1b \
    --bootstrap 1000 \
    --out revisions/outputs/joint/
```

Model set: `revisions/configs/models_fast.yaml` (llama1b, qwen). Scale to all six with
`models_paper.yaml`.

**Outputs** (`revisions/outputs/joint/`)

| file | contents |
|---|---|
| `joint_vs_steering_per_item.csv` | one row per (model, item, condition): `delta_c_steering_published`, `delta_c_steering`, `delta_c_joint`, `qq`, rank-4 ΔC, per-condition trial counts, optional bootstrap CIs |
| `joint_vs_steering_raw_counts.csv` | the raw 2×2 contingency tables behind every ΔC, so anything can be rescored without re-running inference |
| `joint_vs_steering_summary.csv/.tex` | model × condition: mean ΔC per formulation, fraction of items called contextual by each, how often they agree, correlation, mean \|q\| |
| `joint_vs_steering.pdf` | per-item scatter of steering vs joint ΔC, and the paired distributions |

`condition` is `mfirst` / `ffirst` — the option list presented male-first or female-first, the
paper's two conditions. The rank-4 columns treat option order properly as a measurement
setting (CHSH-shaped system) rather than as a filter.

---

## Experiment 2 — Unquantized runs (Reviewer 2, "Quantization")

Re-runs the **published pipeline** with `quantized=False`. `collect_unquantized.py` imports
`generate_one_pronoun` / `generate_one_null_context` and calls them directly, so the items,
prompts, conditions, temperature and `n_runs` are identical by construction rather than by
reimplementation.

```bash
# all six models, primed condition (bump --mem: bf16 weights are much larger)
MODEL=openai/gpt-oss-20b CONDITION=primed sbatch revisions/slurm/collect_unquantized.sh

# compare the two arms
python -m revisions.quantization_compare \
    --models gemma llama8b \
    --quantized   q_gemma.ndjson q_llama8b.ndjson \
    --unquantized u_gemma.ndjson u_llama8b.ndjson \
    --out revisions/outputs/quantization/
```

### The gemma sanity check

`gemma-3-12b-it` was already run unquantized, so re-running it must reproduce the existing
numbers. The two halves of the check have very different expectations, and conflating them
would be a mistake:

* **internal (logit-derived) probabilities are deterministic** given the same weights and
  prompt. They must agree to `--internal-tol` (default 1e-3, absorbing bf16/kernel jitter).
  This is the real test. A failure means the pipeline changed something.
* **generation frequencies are sampled** (temperature 0.5, ~50 draws/cell) and *cannot* match
  exactly. Each item's two counts are compared with a two-proportion test; the fraction of
  items flagged at `--alpha` should land near alpha itself.

```bash
python -m revisions.quantization_compare --sanity-check \
    --new      revisions/outputs/measurements/one_pronoun_measurements_gemma-3-12b-it_0.5_wp_unquant.ndjson \
    --existing one_pronoun_measurements_gemma-3-12b-it_0.5_wp_k40.ndjson
```

It prints `SANITY CHECK PASSED` / `FAILED` and writes every flagged item to
`sanity_check_discrepancies.csv`. (Use the *primed* runs for this: the published null runs
have no usable logits — defect 3 above.)

**Outputs** (`revisions/outputs/quantization/`)

| file | contents |
|---|---|
| `quantization_per_item_raw.csv` | per-item probabilities, counts, KLs and ΔC for each arm separately |
| `quantization_per_item_diff.csv` | the two arms joined, with `diff_*` columns (unquantized − quantized) |
| `quantization_summary.csv/.tex` | per model: mean probability/KL/ΔC in each arm, mean absolute shift, and `frac_verdict_flip_*` — how often the contextual/non-contextual verdict changes when quantization is removed |
| `sanity_check_per_item.csv`, `sanity_check_discrepancies.csv` | the gemma reproduction check |

---

## Experiment 3 — Referent-only null primes (Reviewer 3, "Co-Occurrence")

The contextual prime introduces a pronoun *and* a second mention of the referents. This adds a
prime with the referents but **no pronoun**, separating the two.

Two styles, both generated from the existing WinoPron items:

* **`repeated_np`** (default) — the paired template with its BLANK replaced by the definite NP
  of its own referent. Lexical content and syntax are held constant against the contextual
  prime; the only change is pronoun → antecedent NP. Grammatical case is respected
  (`$POSS_PRONOUN` → `the technician's`).

  ```
  contextual:    The technician told the customer that she had completed the repair.
  referent-only: The technician told the customer that the technician had completed the repair.
  ```

* **`conjunction`** — a minimal frame naming both referents and nothing else, matched in
  spirit to the generic null primes: `The technician and the customer were both present.`

Every generated prime is checked to contain **no personal pronoun** (all 360 pass). 28 of 360
contain an inanimate `it`/`its` inherited from the source template ("found it hard to eat
enough"); these carry no gender, appear identically in the contextual prime, and are recorded
in `has_inanimate_pronoun` rather than treated as failures.

```bash
python -m revisions.primes --style repeated_np          # build the primes (no GPU)
MODEL=google/gemma-3-12b-it sbatch revisions/slurm/collect_referent_null.sh

python -m revisions.referent_null \
    --primed  one_pronoun_measurements_gemma-3-12b-it_0.5_wp_k40.ndjson \
    --null    null_measurements_gemma-3-12b-it_0.5_wp.ndjson \
    --refnull refnull_measurements_gemma-3-12b-it_0.5_wp.ndjson \
    --primes  revisions/outputs/primes/referent_primes_repeated_np.csv \
    --models  gemma \
    --out revisions/outputs/referent_null/
```

KL is the smoothed Bernoulli KL over generation counts against the shared unprimed baseline —
the same estimator as the paper (`(k + 0.5)/(n + 1)`, in bits).

**Reading the result:** if the contextual shift is pronoun priming, `kl_refnull` should look
like `kl_null`. If it is referent co-occurrence, `kl_refnull` should look like the contextual
KLs. `refnull_over_contextual` in the summary is that ratio directly.

### ΔC is *undefined* for the two pronoun-free conditions — and that is the answer

The brief asked for ΔC on all three conditions. For conditions 2 and 3 that quantity does not
exist, and reporting a number would be misleading.

A CbD ΔC needs **two jointly measured content variables per context**. Under a contextual
prime they are the prime pronoun and the generated pronoun. A prime with no pronoun supplies
only one — there is no "prime pronoun" variable — so there is no cyclic system.

You *can* force a number out of the formula by encoding the constant prime as an outcome, but
the answer then depends on that arbitrary choice: on identical counts, calling the pronoun-free
prime "male" gives ΔC = −1.2 and calling it "female" gives ΔC = −1.6, while every measured
quantity is unchanged. (`test_divergences.py::test_delta_c_for_a_pronoun_free_prime_would_be_an_encoding_artifact`
pins this, so the artifact cannot be reintroduced.)

So `delta_c_null` and `delta_c_refnull` are emitted as `NaN` with a `_note` column, and we
report `disturbance_null` / `disturbance_refnull` = `|P(f | forward) − P(f | reverse)|`
instead, which is well defined and invariant.

This is not a hole in the analysis — it is the cleanest possible answer to Reviewer 3: **a
prime containing no pronoun cannot induce contextuality in this design, by construction.** All
it can do is shift the marginal, which is exactly what the KL columns measure. ΔC remains
well defined and is reported for the contextual condition (`delta_c_contextual_{mfirst,ffirst}`).

**Outputs** (`revisions/outputs/referent_null/`)

| file | contents |
|---|---|
| `three_condition_per_item.csv` | per (item, sentence order): counts, P(female) and KL for all three conditions plus the unprimed baseline; ΔC for the contextual condition; disturbance for the pronoun-free ones; the prime text |
| `three_condition_summary.csv/.tex` | per model: mean/median KL by condition, the `refnull_over_contextual` / `null_over_contextual` ratios, and mean disturbance |
| `three_conditions.pdf` | KL distributions by condition |

**Row alignment.** Null-style runs are indexed per *sentence slot*, not per pair: row `2k` is
pair `k`'s sentence-order-`[0,1]` item (free sentence = `template_2`) and row `2k+1` is its
`[1,0]` item. The primes CSV carries `pair_index` and `free_slot` explicitly so the join is on
metadata, not on position (`test_primes.py::test_row_order_and_sent_order_mapping` pins it).

---

## Experiment 4 — Per-item plain probability table (Reviewer 2, "Plain Probabilities")

Figure 1a–e aggregates over items; this emits the per-item numbers underneath it, as CSV and
as a LaTeX `longtable` for the supplement.

```bash
python -m revisions.per_item_table \
    --measurements one_pronoun_measurements_gemma-3-12b-it_0.5_wp_k40.ndjson \
                   one_pronoun_measurements_phi-4_0.5_wp_k40.ndjson \
    --models gemma phi \
    --out revisions/outputs/per_item/
```

One row per (model, item, sentence order) — the unit Figure 1 plots as a single marker:
P(female) unprimed / m-primed / f-primed with Wilson 95% intervals and the raw `k`/`n`; the
plain differences `shift_m_primed`, `shift_f_primed` and `spread`; the smoothed and internal
(logit) versions of each; and `ordered`, whether the expected monotone pattern
`P(f|m) ≤ P(f|∅) ≤ P(f|f)` holds.

As a check, the mean shifts are compared against the published aggregator
(`analysis.mean_completion_shift`, imported and run unmodified) and a warning is logged on any
disagreement.

**Outputs:** `per_item_probabilities.csv`, `per_item_probabilities_<model>.tex`.

---

## Reproducibility

* `--seed` (default 20260713) seeds `random`, `numpy` and `torch`, and is recorded in every
  sidecar. Note that generation is still sampled at temperature 0.5 — the seed makes a run
  reproducible, it does not make two different runs identical.
* Every results file has a `.meta.json` with git commit, library versions, model revision hash
  (resolved from the HF Hub when reachable), quantization setting and argv.
* Raw per-item outputs (contingency tables, counts) are written next to every aggregate, so
  any analysis can be rescored without re-running inference.

## Notes on the environment

* `gpt-oss-20b` needs `transformers >= 4.55` (`run_local.py` imports `Mxfp4Config`). The
  existing `requirements.txt` pins `transformers` unpinned; it was left alone rather than
  constrained, per the no-edit rule.
* The collection scripts require CUDA (`config.GPU_INDEX = 'cuda:0'`). The analysis CLIs, the
  prime builder and the test suite run anywhere — collection modules import `ModelProbs`
  lazily so `--help`, `--dry-run` and `pytest` work without a GPU.
* `--dry-run` on the collectors builds every prompt and record without loading weights, which
  is the cheap way to check a prompt change before burning GPU hours.
