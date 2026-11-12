# AGENTS.md

Instructions for coding agents working in this repository. Humans should start with [README.md](README.md).

## What this is

`rsabias` is the research tool behind "Biased RSA private keys: Origin attribution of GCD-factorable keys" (ESORICS 2020, https://arxiv.org/abs/2009.06700). It estimates which library, smartcard or HSM generated an RSA **private** key, using a pre-trained Bayes model over 26 groups of sources. It also has the pipeline that built that model from a dataset of over 150 million keys.

## Which instructions to read

| Goal | Read |
|---|---|
| Classify or analyse someone's RSA keys | [.claude/agents/rsa-key-analyst.md](.claude/agents/rsa-key-analyst.md): background, running the classifier, reading and combining reports, how to report results |
| Get keys into the format the tool accepts | [.claude/skills/prepare-rsa-keys/SKILL.md](.claude/skills/prepare-rsa-keys/SKILL.md) and its `scripts/normalize_keys.py` |
| CLI tasks and parameters, supported key formats | [README.md](README.md) |
| Model, groups, features, distributions | [rsabias/model/readme.md](rsabias/model/readme.md) |
| Report table generation | [rsabias/classification_table_template/](rsabias/classification_table_template) |

In Claude Code, `rsa-key-analyst` is a subagent that has the `prepare-rsa-keys` skill preloaded. Other agents can read both files as plain instructions. The script only needs Python 3.7+ and pycryptodome.

## Rules for any work involving keys

- Private keys are secrets. Never print or paste key material, commit keys or key-derived reports, or send keys to external services. Refer to keys by file name or by the SHA-256 of the modulus.
- Work on copies outside the repository. Never modify the user's original key files.
- Never reorder the primes. The features read p and q separately, so swapping them changes the classification.

## Layout

- `rsabias/cli.py`: the CLI (`rsabias -a <task> ...`). It dispatches to `rsabias/core/tasks.py`.
- `rsabias/core/`:
  - `key.py`: key parsing;
  - `features.py`: feature transformations;
  - `classifier.py`: the `ProbaTable`, `Classifier` and `Evaluator` classes;
  - `dataset.py`: dataset readers and writers with `meta.json` descriptors.
- `rsabias/model/`: the shipped model (`classification_table_complex.npz`; the `.pkl` original only loads with the 2020 dependencies), `groups/groups.json`, `precision_recall.json`, `transformations/`, `distributions/` (the per-source distributions the model was built from).
- `datasets/baby_dataset/`: small labelled sample keys, gzipped JSON lines with hex values.
- `datasets/BatchGCD-primes-only/`: single primes from GCD-factorable TLS keys.
- `tests/`: golden tests that compare the tasks with outputs recorded in the 2020 environment.
- `tools/`: `summarize_classification.py` (combines `classify` reports of keys from one source), `pem2dataset.py` (PEM keys to a dataset with `meta.json`), `compare_dist.py` (compares two `dist.json`), `convert_model.py` (pickled classification tables to `.npz`).
- `docs/guide.md`: walkthroughs for testing one's own key generation; `work/` (git-ignored) is its working directory for keys and outputs.
- `pyproject.toml`, `uv.lock`, `Dockerfile`: packaging for Python 3.11+ with current dependencies.

## Running

- Local: `pip install .` (Python 3.11+) or `uv sync`, then `rsabias -a <task> ...`. Check with `rsabias -h`.
- Docker: `docker images rsabias` shows whether the image is built (`docker build -t rsabias .`). Its entrypoint is `rsabias` and its working directory `/work`. Mount inputs and outputs, pass `--user "$(id -u):$(id -g)"`, and create output directories on the host first, otherwise Docker creates them owned by root and the tool cannot write into them.
- The environment of the paper (Python 3.7, the 2020 dependencies) is built from the tag `esorics2020-env`, see the README. It is only needed to reproduce results bit for bit or to regenerate the golden tests.

## Changing code

Run `uv run pytest` (add `-m "not slow"` to skip the pipeline tests). The golden outputs in `tests/golden` must not be regenerated in the current environment: they are the reference that shows the current dependencies reproduce the 2020 results.

## Known pitfalls in the current code

- `classify` opens the key file as text, so DER input fails. It doesn't take a passphrase, so encrypted keys fail too.
- OpenSSH private keys are accepted but come out with p and q swapped (pycryptodome's importer swaps them), which gives a wrong group without any error.
- `classify` writes `key_classification.json` into `-o` and overwrites it on every run.
- `evaluate` ignores `-l` and always runs the labelled accuracy evaluation. There is no CLI path for classifying an unlabelled batch.
- `batch_gcd` reads a hardcoded placeholder path (`see/comment/above` in `Evaluator.init_batch_dictionary`).
- The cross-feature classifier described in the paper cannot be selected from the CLI (`-m` accepts `complex` or `naive`).
- `classify`, `evaluate` and `group` fail with a traceback when the `-o` directory does not exist (`dist` creates it).
- The model only knows sources collected until 2020. Keys from newer libraries are still assigned a group; OpenSSL 3, for example, does not land in the OpenSSL group 24.
