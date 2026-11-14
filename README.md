# EMRLL
# EMRLL: Mathematical-Answer Generation Experiments

> Publication preparation: replace every `[COMPLETE: ...]` field and remove this
> notice before publishing. The inventory must describe the files actually
> deposited, not the files intended for a future release.

## 1. Release information

| Field | Value |
|---|---|
| Version | v0.1.0 — use the actual release tag if different |
| Release date | [COMPLETE: actual release date, YYYY-MM-DD] |
| Software/archive DOI | [COMPLETE: this software release's reserved or assigned DOI] |
| Source repository | [COMPLETE: repository URL, or state that no public repository is provided] |
| Archived commit | [COMPLETE: commit identifier, or “not applicable — direct deposit”] |
| Status | Initial research snapshot; not a certification of full experimental reproducibility |

Associated manuscript: **Resource-Constrained Mathematical-Answer Generation:
A Design Framework and Evaluation Audit**.

This software release is distinct from the manuscript. No journal acceptance
or publication is claimed here. A manuscript identifier should be added
separately when available.

## 2. Author and contact

**Mohammed Anouar Naoui** — sole author and corresponding author

Artificial Intelligence and Applications Laboratory  
Department of Computer Science  
University of Echahid Hamma Lakhdar El-Oued  
El-Oued, Algeria

ORCID: https://orcid.org/0000-0003-1653-531X  
Email: naoui-anouarmohammed@univ-eloued.dz

## 3. Project overview

EMRLL investigates mathematical-answer generation under constrained
computational resources. Its experimental program examines unadapted model
references, supervised fine-tuning, LoRA/QLoRA adaptation, GRPO-based training,
and a quadratic-data intervention.

The models investigated are GPT-2, DeepSeek-R1-Distill-Qwen-1.5B,
Qwen2.5-7B, and DeepSeek-R1-Distill-Qwen-14B. The evaluation categories are
arithmetic, linear equations, multi-step problems, quadratic equations,
and word problems.

The project emphasizes category-level outcomes, evaluation sensitivity,
and traceable experiment provenance. It does not claim that one adaptation
method is universally superior or that the available scores isolate model
size, numerical precision, or adaptation algorithm as a single causal factor.

## 4. Files included and access conditions

Complete the inventory below using exact paths within this archive. For every
component, state whether it is included, available elsewhere, restricted,
or unavailable. Listing an experiment in this README does not establish that
all of its artifacts are included.

| Component | Archived path or access status |
|---|---|
| Experiment source scripts | [COMPLETE: paths and experiment coverage] |
| Configurations and environment records | [COMPLETE: paths and whether they are executed-run records or current snapshots] |
| Training, validation, and evaluation data | [COMPLETE: filenames, versions/hashes, and access conditions] |
| Raw model completions and historical scores | [COMPLETE: paths, or explain their absence] |
| Scoring-audit code and output | [COMPLETE: paths and audit version] |
| Reproduction instructions | [COMPLETE: path to tested installation and execution instructions] |
| Manuscript and supplementary documentation | [COMPLETE: paths, or state that they are not included] |
| Fine-tuned checkpoints/adapters | [COMPLETE: paths or separate access conditions; do not imply they are included if absent] |

The original historical E02 corpus and checkpoint were not recovered. The
E02-R corpus was constructed by sampling the available E04 training corpus
and splitting it into training and validation subsets. It is not claimed to
be identical to the historical E02 corpus.

Base models should be identified by their upstream repository and revision
in the experiment configuration. Their weights, licenses, and access
conditions are separate from this project's code release.

## 5. Reported experiment outcomes and scoring sensitivity

The historical column below transcribes the supplied experiment summaries.
The strict-audit column transcribes the local audit report
`SNCS_20261114T082712Z_770485`, executed on 14 November 2026.
Each scored evaluation contains 250 examples, with 50 per category.

**These columns use different scoring procedures. Strict-audit scores do not
automatically replace historical scores or establish corrected mathematical
accuracy.**

| Run | Configuration label | Historical correct / 250 (%) | Strict-audit correct / 250 (%) |
|---|---|---:|---:|
| E01 | GPT-2 reference | 1 (0.4%) | 0 (0.0%) |
| E02-R0 | Initial GPT-2 supervised reproduction | 0 (0.0%) | 0 (0.0%) |
| E02-R1 | Answer-focused/EOS FP16 smoke diagnostic | Not evaluated | Not evaluated |
| E02-R2 | GPT-2 reproduction labelled R2 | 1 (0.4%) | 0 (0.0%) |
| E03 | Later DeepSeek 1.5B reference | 35 (14.0%) | 29 (11.6%) |
| E04 v1 | DeepSeek 1.5B GRPO-labelled configuration | 212 (84.8%) | 196 (78.4%) |
| E04 v2 | Quadratic-data intervention | 198 (79.2%) | 174 (69.6%) |
| E05 | Qwen2.5-7B reference | 216 (86.4%) | 61 (24.4%) |
| E06 | Qwen2.5-7B QLoRA | 147 (58.8%) | 46 (18.4%) |
| E07 | DeepSeek 14B reference | 161 (64.4%) | 162 (64.8%) |
| E08 | DeepSeek 14B QLoRA | 194 (77.6%) | 170 (68.0%) |
| E04-earlier-base | Earlier reference evaluated alongside E04 | 45 (18.0%) | 43 (17.2%) |

E02-R1 showed non-finite gradient diagnostics in its recorded smoke test;
no completed holdout accuracy is assigned to that run. The intended E02-R2
changes were answer-focused supervision, explicit EOS, and BF16, but the
executed objective, collated labels, and checkpoint provenance remain to
be established from the corresponding artifacts.

The earlier E04-associated reference is distinct from the later E03 run.
They must not be substituted for one another without checking inference
and scoring settings.

The audit performed no training or new model generation. Its
`explicit-atomic-v1` parser selects the last boxed field, otherwise the last
explicit answer/solution field, otherwise a wholly numeric completion.
It requires a supported finite numerical literal and does not fall back to
the last number anywhere in the reasoning text. This narrower rule can reject
otherwise valid answer formats. Manual examination of disagreements is needed
to distinguish rejected correct answers, historical false positives, and
ambiguous or incomplete outputs.

## 6. Data interpretation

The original experiment CSV schema uses `id`, `category`, `problem`, and
`answer`. Prediction CSV files additionally record fields such as
`prediction`, `correct`, and `generated`. Document any release-specific
column changes in the relevant file's accompanying description.

A historical `correct` value belongs to its original scoring procedure.
Do not silently replace it with a post-hoc parser's correctness value.
Align paired evaluations using identifiers and verify the problem text,
category, and reference answer; do not assume that row order is sufficient.

The 250-example evaluation set was repeatedly inspected during development.
It should not be described as an untouched confirmatory test. Balanced
category counts do not establish independence of mathematical instances
across training, validation, and evaluation splits.

## 7. Environment and use

The supplied local run logs report Linux, Python 3.10.21,
PyTorch 2.11.0+cu128, and an NVIDIA RTX A5000 with approximately 23.55 GiB
of reported GPU memory. Experiment scripts use libraries including
Transformers, Datasets, Accelerate, PEFT, TRL, and bitsandbytes where relevant.

Use the release's environment record for actual package versions. A current
environment snapshot is not proof of the environment used by every historical
run. A clean-environment, end-to-end reproduction is not certified by this
README.

Before execution, check the selected script's input files, model revision,
prompt, precision, generation settings, scorer, and output location. Historical
scripts may contain absolute HPC paths; adapt a separate copy and record the
changes. Do not overwrite existing models, logs, or scores.

### Optional verification toolkit

The following command applies only if the supplied
`EMRLL_SNCS_Verification/` directory is included unchanged in this release:

```bash
python3 EMRLL_SNCS_Verification/test_audit.py
```

This runs synthetic software tests, not model experiments. Its success does
not certify scientific correctness. Consult the toolkit's own README before
running the local project audit; that audit requires existing experiment
files and does not recreate missing checkpoints or generate new responses.

## 8. Known limitations and provenance

The remaining verification questions include the identity of executed versus
current scripts, the exact E04 reward and complete GRPO settings, E02-R2's
post-collation supervision and checkpoint identity, and symbolic-instance or
template overlap between data splits.

Prompt wording, loading precision/quantization, stopping behavior, and
numerical extraction differ across parts of the experimental history.
Rescoring existing completions cannot repair these generation differences.
An untouched confirmatory test and additional matched comparisons remain
future validation steps, not completed findings of this release.

## 9. License and third-party materials

Software license: [COMPLETE: selected license and license-file path].

Data/documentation terms: [COMPLETE: applicable licenses, restrictions,
and any excluded materials].

Retain applicable third-party notices. Do not assume that the software license
also licenses external model weights, third-party datasets, or manuscript
material. Match the license metadata on the archival record to the terms
actually supplied with the deposited files.

## 10. Funding, competing interests, and AI assistance

The author received no funding for this research.

The author declares no financial or non-financial competing interests
relevant to this work.

ChatGPT (OpenAI) assisted with manuscript planning, drafting and revision;
literature-search suggestions and synthesis; proposition development;
LaTeX and submission-material preparation; development and debugging of
training, evaluation, and statistical-analysis code; and reproducibility and
answer-extraction audit design. This assistance extended beyond copy-editing.

AI-generated material is not independent evidence of experimental validity
and does not authenticate historical runs or replace missing artifacts.
Mohammed Anouar Naoui is the sole author and retains responsibility for
scientific judgments, accuracy, integrity, attribution, and final approval.
No AI tool is credited as an author.

## 11. Citation

Cite the specific release used in your work. Complete the following citation
with the actual release metadata before publication:

Naoui, Mohammed Anouar. [COMPLETE: release year].
EMRLL: Mathematical-Answer Generation Experiments.
Version v0.1.0. Zenodo. [COMPLETE: version-specific software DOI].

The software/archive DOI and any later journal-article DOI identify different
research objects. Record them separately. For questions or reports of problems,
contact the author using the email above and identify the release version,
experiment, and affected file.
