# EMNLP revision

Code for the sensitivity analyses requested in review, and a record of defects found in the
original analysis code while preparing them.

```
revisions/
  cbd.py      independent CbD implementation, used to cross-check the package's ΔC functions
  tests/      pytest suite for the CbD fixes: `pytest revisions/tests` (no GPU or data needed)
```

---

## Read this first: defects found in the existing code

Defects 1, 2 and 4–6 are FIXED in `winogender_contextuality/modeling/contextuality.py`;
defect 3 is fixed in `collect_sequential.py` for future runs, but the null files already
collected keep their bad logits; the remaining defects are FIXED in the paper notebook. Most of
them affect published numbers, so all of them need a decision before the response letter goes
out.

Every fixed function is cross-checked against `revisions/cbd.py`, an independent implementation
written from the CbD definitions rather than from the original code
(`revisions/tests/test_package_cbd_fixes.py`).

**Bottom line: every ΔC in the paper needs re-running.**

### 1. The steering ΔC used the wrong marginal — FIXED (this changes reported ΔC)

`calculate_sentence_dc_fraction` built the forward correlation as

```python
cbd_correlation(V1, V2, V1W2)     # V1 = P(prime=f | fwd),  V2 = P(prime=f | rev)
```

CbD calls for the two marginals **of that context** — the prime marginal and the *generation*
marginal, `(V1, W2)`. `V2` is the prime marginal of the *other* context. (The reverse term and
the inconsistent-connectedness term were already correct.)

Because primes are balanced by design (`V1 = V2 = 0.5`), the old forward correlation was
offset by exactly `E(W2) = 2·W2 − 1` — **the model's deviation from a 50/50 female-generation
rate.** In a gender-bias experiment that deviation is systematically nonzero, so the error does
not average out. On randomly drawn admissible tables the mean |ΔC shift| is ≈ 0.40 and the
contextual/non-contextual verdict flips on ≈ 14% of items.

**You must re-run the ΔC numbers.** Every ΔC in the paper computed with this function is
affected, in a direction that correlates with the very bias being measured.

### 2. `mode='internal'` ΔC had an identically-zero joint probability — FIXED

The internal branch computed its joint count as

```python
forward_trials = zip(data_dict['forward']['fixed_pnoun'],   # strings: 'he' / 'she'
                     data_dict['forward']['free_pnoun'])    # LISTS of logits
count_c1 = sum(1 for x, y in forward_trials if x == target_f[0] and y == target_f[1])
```

`y` is a logit vector and `target_f[1]` is a pronoun string, so the comparison was never true
and `V1W2 == 0` for every item: internal-mode ΔC was not measuring a joint distribution at all.
It now uses each trial's own logits, `p(x,y) = p(y|x)·p(x)`:

```python
V1W2 = mean_t [ 1{prime_t = female} · softmax(logits_t)[1] ]
```

The forward marginal `W2` also changed, from `softmax(mean(logits))` to `mean(softmax(logits))`.
The old form averages in *logit* space across two different prompts (male-primed and
female-primed), which is not the marginal of the 50/50 prime mixture; the new form is, and it
makes the marginal and joint mutually consistent. **Any internal-mode ΔC in the paper must be
re-run.** Generation-mode numbers were unaffected by defect 2.

While fixing this I also corrected the reverse context's joint count, which had its prime and
free targets transposed. It was harmless in practice — both sentences of a pair always share a
grammatical case, so the two target strings coincide — but it was load-bearing on that
coincidence, and now it is not.

### 3. The null runs store logits over the characters of "she" — FIXED for future runs only

`collect_sequential.generate_one_null_context` does:

```python
pronouns = ast.literal_eval(df.differences[idx])   # a LIST: ['he', 'she']
pronoun_idxs = mp.get_token_ids(options=pronouns[1])   # pronouns[1] is the STRING 'she'
```

`get_token_ids` iterates its argument, so it tokenises `'s'`, `'h'`, `'e'` and the `logits`
field of every existing `null_measurements_*.ndjson` is a 3-vector over those characters, not
a 2-vector over `['he','she']`. (In `generate_one_pronoun`, `pronouns` is a *dict*, so
`pronouns[1]` is at least an option list; see the note below.)

The published null analysis uses generation counts, so this defect does not touch it. No
logit-based null quantity can be computed from the existing files. `generate_one_null_context`
now passes the whole option list, so null runs collected from now on store a proper 2-vector
over `['he','she']`.

`generate_one_pronoun` (and `generate_one_pronoun_noprompt`) had a milder version: they took
logits for `pronouns[1]`, the options of `template_2`, even when the free sentence was
`template_1`. Every WinoPron pair uses the same case in both sentences, so the published logits
are unaffected, but they now use the free sentence's own options (`p2`).

### 4. `calculate_sentence_nc_fraction` had the identical `(V1, V2)` defect — FIXED

The two-pronoun sibling of defect 1, in the same file: its forward correlation used `V2` (the
sentence-2 marginal of the *reverse* context) in place of `W2`. Same error, same consequences.
(The reverse term was already correct — `cbd_correlation` is symmetric in its first two
arguments.)

### 5. `cbd_s1_4cycle` maximised over an inadmissible sign pattern — FIXED

CbD's `s1` maximises only over sign vectors with an **odd** number of minus signs. The term
`|w + x - y - z|` has two. It was standing in for the admissible `|w + x - y + z|`, so the
function made **two errors at once**:

| | frequency on random correlation vectors |
|---|---|
| included the inadmissible `\|w+x-y-z\|` → **overstated** s1 | 12.6% |
| dropped the admissible `\|w+x-y+z\|` → **understated** s1 | 21.9% |
| exactly right | 65.5% |

Mean \|error\| when wrong: 0.555. Since `ΔC = s1 − 2 − disturbance`, the old version could both
**report contextuality where there is none** and **miss real contextuality**. This feeds
`calculate_pronouns_nc_fraction` and `measure_contextuality.py`.

### 6. The rank-4 system was not walked in cycle order — FIXED (in the right place)

A rank-4 cyclic system requires *consecutive contexts to share a content variable*.
`calculate_pronouns_nc_fraction` consumed its four contexts in the order
`(0,0), (0,1), (1,0), (1,1)`, which places `(0,1)` next to `(1,0)` — two contexts that share
none — so the disturbance terms were paired with the wrong contexts.

**The fix belongs in the CbD function, not in `pronoun_context_array`.** That array's product
ordering is *correct* for the sheaf/LP code: `MeasurementScenario.incidence_matrix` is built
consistently with it, and `check_feasibility` / `calculate_contextual_fraction_abramsky` depend
on it. Reordering the array would have silently broken the contextual-fraction computation. So
`calculate_pronouns_nc_fraction` now reindexes into cycle order `(0,0), (1,0), (1,1), (0,1)`
internally, and swaps the two marginals in the contexts where the lower-indexed content variable
is the sentence-2 pronoun. (The correlation is symmetric in its marginals, so only the
disturbance pairing changes.)

### 7. The null divergence bars showed P(female), not KL — FIXED

`get_null_divergences` in the paper notebook computed `null1_kl` / `null2_kl` but appended
`null1_prob` / `null2_prob`, so the `c_null` bars in the divergence figures were mean
P(female | null prime), plotted next to the KL divergences of `c_m` / `c_f`. It now returns KLs
in all four columns. **Re-run the divergence figures.**

### 8. The notebook paired rows by position across two different orders — FIXED

Two orderings live side by side in `notebooks/13-sk-paper-figures.ipynb`:

* **item order**, which `primed_completion_differences()` / `primed_completion_difference_counts()`
  emit and `create_feature_table()`'s `2*index` / `2*index + 1` lookup assumes: pair `k`
  contributes sentence order `[0,1]` (free sentence = `template_2`) and then `[1,0]`
  (free = `template_1`);
* **sentence order**, `template_1` then `template_2`: `all_sentences_wp.csv`, and therefore every
  null run, and `new_templates.tsv`, and therefore `all_features`.

Wherever the two were zipped together by position, every sentence was matched with the *other*
sentence of its pair. The notebook now joins on sentence text instead (`primed_item_keys` plus
one aligner per table), and `get_null_divergences` refuses input that has not been aligned.

* **Null divergences** compared each null prime against the unprimed baseline of the other
  sentence (`align_null_counts`). With synthetic responses on the real templates and a null
  prime that has no effect, the aligned null KL is ~0; the old positional pairing gives 0.58 bits.
* **Stereotype correlations** correlated each sentence's P(female) with the stereotype of its
  pair-mate's referent, because `all_features` was in sentence order. It is now reordered into
  item order right after it is built (`align_features_to_items`).
* **Mutual information** reads the same table through `create_feature_table`. For role type the
  swap is only a relabelling, which MI is invariant to, but rows with a missing stereotype are
  dropped *before* MI is computed, so the wrong sentence's measurements were dropped whenever
  exactly one referent of a pair had no stereotype. Fixed by the same reordering: every feature
  row's role is now the referent of its item's free sentence (before: 0/360).

**Re-run the divergence, stereotype-correlation and MI figures.**

### 9. The temperature stereotype correlations read the wrong runs — FIXED

The three cells that loop over `gpt_temps` indexed `all_gen_diffs[n]`, i.e. gpt, phi and gemma
at T = 0.5, instead of `temp_gen_diffs[n]`. **Re-run the temperature stereotype-correlation
figures.**

### Still outstanding — not a CbD defect, so left alone

* `generate_two_pronouns` constructs `Context` without `case_1`/`case_2`, which are required
  fields, so it raises `TypeError` before writing a record. It is the natural collector for the
  joint-measurement experiment, so it gets fixed there.
