# Biased RSA private keys: Origin attribution of GCD-factorable keys

Python tool for black-box analysis of RSA key generation in cryptographic libraries and RSA key classification. This tool accompanies the paper [Biased RSA private keys: Origin attribution of GCD-factorable keys](TBA) presented at [ESORICS 2020](https://www.surrey.ac.uk/esorics-2020) conference.

Using this project, you can audit the origin of your RSA private key. Also, it is possible to replicate our work and analyze large datasets of keys. Beware that when classifying one key, the process is not bulletproof, as our model achieves 47.7% accuracy (on 26 classes) on average.

What the tool can and cannot tell:

- It needs **private** keys: the features are computed from the primes `p` and `q`.
- The answer is one of **26 groups** of sources whose keys are indistinguishable, not a single library version ([groups.json](rsabias/model/groups/groups.json)).
- One key is classified correctly in 47.7% of cases on average; 10 keys from one source in about 89%.
- The model knows the 70 sources collected until 2020. Keys of other or newer implementations are still assigned the closest group; OpenSSL 3, for example, does not land in the OpenSSL group 24. Comparing distributions ([Comparing key sets](#comparing-key-sets)) shows such differences directly.

For a step-by-step introduction to testing your own key generation, see the [guide](docs/guide.md).

## Install

The tool requires Python 3.11 or newer. Since the stable part of the code is tied to the experimental part, this package is not distributed on PyPI; install it from a clone:

```
git clone git@github.com:crocs-muni/RSABias.git &&
cd RSABias &&
python3 -m venv venv &&
source venv/bin/activate &&
pip install .
```

### Docker

The image contains the tool with the exact dependency versions from `uv.lock` and the model. Mount the directory with your keys as `/work`, the working directory of the image; outputs are written there too:

```
docker build -t rsabias --build-arg VCS_REF=$(git rev-parse --short HEAD) .
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD:/work" rsabias -a classify -i key.pem -o .
```

`--user` makes the container read and write the mounted files as you; without it, it runs as UID 1000.

### Development and tests

[uv](https://docs.astral.sh/uv/) installs the exact dependency versions from `uv.lock` together with the test tools: `uv sync`, then `uv run pytest` (add `-m "not slow"` to skip the model-building tests, which take a few minutes).

The tests compare the tasks (`classify`, `dist`, `group`, `build`, `evaluate`) on the keys in `tests/data` and the [baby dataset](datasets/baby_dataset) against golden outputs in `tests/golden`, recorded in the legacy environment of the paper (below). They therefore check that the current dependencies reproduce the original results.

### Legacy environment (ESORICS 2020)

The original environment of the paper (Python 3.7, dependency versions from 2020) is available as a Docker image. It produced the golden outputs the regression tests in `tests/` compare against. The current code requires a newer Python, so the image is built from the tag `esorics2020-env`, the last commit that runs in it:

```
git archive esorics2020-env | docker build -t rsabias:legacy --build-arg VCS_REF=esorics2020-env -
docker run --rm -v "$PWD/out:/out" rsabias:legacy -a classify -i rsabias/model/sample_private_key.pem -o /out
```

## Usage

### Classification of a single key

From the repository root, you can try to classify [the example key](https://github.com/crocs-muni/RSABias/blob/master/rsabias/model/sample_private_key.pem) as:

```bash
rsabias -a classify -i rsabias/model/sample_private_key.pem -o .
```

where `-i` parameter denotes a path to the RSA private key in PEM format, and `-o` parameter denotes a path to the folder where the classification report will be stored. Consequently, running the command above should print basic information to the standard output:

```bash
Your key with sha-256 digest: e10d8b68b1e2b595f69813a99524d9dce0c0f80b092f9466d80b587f587262ba gets classified as:
	- Classification group: 24
	- Score: 0.023450350877192983
	- If a key is marked as coming from Group 24, the model is in 90.21% cases right.
	- The model correctly classifies 96.83% keys that are actually from Group 24.
	- This group contains the following sources (and your key therefore probably originates from):
		* Library OpenSSL 1.0.2g 2.0.12 fips 1024
		* Library OpenSSL 1.0.2g 2.0.12 fips 2048
		* Library OpenSSL 1.0.2g 2.0.12 fips 512
		* Library OpenSSL 0.9.7 1024
		* Library OpenSSL 0.9.7 2048
		* Library OpenSSL 0.9.7 512
		* Library OpenSSL 1.0.2k 2.0.14 fips 1024
		* Library OpenSSL 1.0.2k 2.0.14 fips 2048
		* Library OpenSSL 1.0.2k 2.0.14 fips 512
		* Library OpenSSL 1.1.0e 1024
		* Library OpenSSL 1.1.0e 2048
		* Library OpenSSL 1.1.0e 512
		* Library OpenSSL 1.0.2g 1024
		* Library OpenSSL 1.0.2g 2048
		* Library OpenSSL 1.0.2g 512
	- The full report of the key classification can be found at: ./key_classification.json
```

The full report at `./key_classification.json` then contains similar information for all groups. At the bottom of the report, one can usually see so-called impossible groups. Such groups from our experience cannot generate the examined key. At the very top, the most probable groups are listed.

#### Supported key formats

The key file must hold exactly one RSA **private** key, **without a passphrase**, in **PEM** format. The classifier needs the primes `p` and `q`, so public keys and certificates cannot be classified.

| Key file | Header | Supported |
|---|---|---|
| PKCS#1 PEM | `-----BEGIN RSA PRIVATE KEY-----` | Yes |
| PKCS#8 PEM | `-----BEGIN PRIVATE KEY-----` | Yes |
| DER (binary, PKCS#1 or PKCS#8) | – | No, the file is read as text and fails with `Failed to read key-data from file, 'utf-8' codec can't decode byte ...` |
| Passphrase-protected PEM | `Proc-Type: 4,ENCRYPTED` or `-----BEGIN ENCRYPTED PRIVATE KEY-----` | No, fails with `Failed to build a key object from the key, The private key is encrypted: a passphrase is required.` |
| Public key | `-----BEGIN PUBLIC KEY-----` | No, fails with `Failed to build a key object from the key, No CRT component 'p' available for public keys.` |
| OpenSSH | `-----BEGIN OPENSSH PRIVATE KEY-----` | Do not use, see below |

An OpenSSH private key is parsed without an error, but `p` and `q` come out swapped compared with the PEM encoding of the same key. The features treat `p` and `q` differently, so the key can end up in a different group (the example key above falls into group 6 instead of group 24). Convert such keys to PEM before classifying them.

Keys in other encodings can be converted with OpenSSL or OpenSSH:

```bash
openssl rsa -inform DER -in key.der -out key.pem        # DER -> PEM
openssl rsa -in encrypted.pem -out key.pem              # remove the passphrase
cp id_rsa key.pem && ssh-keygen -p -m PEM -N "" -f key.pem   # OpenSSH -> PEM (rewrites the copy in place)
```

The printed sha-256 digest is computed from the contents of the key file, not from the key itself. The same key stored as PKCS#1 and as PKCS#8 therefore gets two different digests.

Only one key is classified per invocation. The `evaluate` task cannot currently be used to classify an unlabelled set of your own keys, see below.

### Classifying a set of keys

Keys that come from the same source can be classified together, which is much more reliable than a single key. Classify every key, then combine the reports with [tools/summarize_classification.py](tools/summarize_classification.py), which multiplies the scores of each group over all keys as `evaluate` does for batches:

```bash
for f in keys/*.pem; do n=$(basename "$f" .pem); mkdir -p "reports/$n"; rsabias -a classify -i "$f" -o "reports/$n"; done
python tools/summarize_classification.py reports/*/key_classification.json
```

### Your keys as a dataset

The dataset tasks (`dist`, `group`, `build`, `evaluate`) read directories with a `meta.json` that describes the source and lists the key files. [tools/pem2dataset.py](tools/pem2dataset.py) creates one from PEM files; `--group` sets the group the keys are assumed to come from, which `evaluate` needs:

```bash
python tools/pem2dataset.py mykeys/openssl3 OpenSSL 3.5.3 keys/*.pem --group 24
```

Without the tool, concatenate the PEM files into one file and write the `meta.json` yourself, with `"format": "pem"` and the number of keys in `records` (`evaluate` needs it):

```json
{
  "type": "reference",
  "files": [{"name": "keys.pem", "records": 30}],
  "details": {"category": "Library", "name": "OpenSSL", "version": "3.5.3", "bitlen": 2048,
              "format": "pem", "compressed": false, "fips_mode": false, "group": "24"}
}
```

The other supported formats (JSON lines, CSV, ...) and the `meta.json` fields are described with the [baby dataset](datasets/baby_dataset/readme.md).

### Comparing key sets

`dist` counts how often every combination of feature values occurs in a dataset (about 5 minutes per million keys) and writes a `dist.json` per source. [tools/compare_dist.py](tools/compare_dist.py) compares two of them: per feature the share of every value, the difference in percentage points and a chi-square test, then the combinations that differ most. Either side can be a directory, whose distributions are summed, e.g. a library in [rsabias/model/distributions](rsabias/model/distributions), which holds the distributions of all sources of the model:

```bash
rsabias -a dist -i mykeys -o dist_out -t rsabias/model/transformations/5p_5q_blum_mod_roca.json
python tools/compare_dist.py rsabias/model/distributions/Library/OpenSSL/1.1.0e/2048/dist.json dist_out
```

To see which known sources are closest, run `group` on your distributions together with a copy of `rsabias/model/distributions`; see the [guide](docs/guide.md#6-walkthrough-3-compare-key-sets).

### Evaluating keys against a group

`evaluate` measures how well the model recognises the keys of each dataset as the group named in its `meta.json`, alone and in batches of 10, 20, 30 and 100 keys of one source. Results are summed per group, not per dataset, and every key is classified once per batch size (about 3.5 ms each):

```bash
mkdir -p eval_out
rsabias -a evaluate -i mykeys -o eval_out -t rsabias/model/transformations/5p_5q_blum_mod_roca.json -g rsabias/model/groups/groups.json -c rsabias/model
```

It writes `n_<batch size>_results.json`, which counts per true group whether it came first, among the first two or among the first three (`{"n_keys": ..., "accuracies": {"1"|"2"|"3": {group: {"correct", "wrong"}}}}`), and `confusion_matrix.{txt,png,pkl}` for single keys.

[scrutiny-viz](https://github.com/crocs-muni/scrutiny-viz) reads this directory and compares two evaluations, e.g. a reference source of the group against your keys, in an HTML report. From its directory:

```bash
python scrutinize.py map -t rsabias --folder /path/to/eval_reference -o results/reference.json
python scrutinize.py map -t rsabias --folder /path/to/eval_out -o results/mine.json
python scrutinize.py verify -s scrutiny/schemas/RSABiasEval.yml -r results/reference.json -p results/mine.json -o results/verify.json
python scrutinize.py report -p results/verify.json -o mine.html
```

The report currently sums the top-1, top-2 and top-3 counts into one accuracy, labels the confusion matrix heatmap by matrix position instead of group number, and is written into scrutiny-viz's `results/` directory regardless of `-o`.

### Tools

| Script | Purpose |
|---|---|
| [tools/summarize_classification.py](tools/summarize_classification.py) | Combines the `classify` reports of keys from one source |
| [tools/pem2dataset.py](tools/pem2dataset.py) | Packs PEM private keys into a dataset with a `meta.json` |
| [tools/compare_dist.py](tools/compare_dist.py) | Compares two `dist.json` files, or directories of them |
| [tools/convert_model.py](tools/convert_model.py) | Converts a pickled classification table of 2020 to `.npz` (legacy environment only) |

The scripts run with the dependencies of the tool; in Docker, use `--entrypoint python`.

### Advanced usage

A bunch of distinct tasks is available for full analysis of the RSA keys. This section of the readme is not meant to be self-contained, but rather provide guidance. The user will most probably need to dive into the code when aiming for the replication of the experiment. A list of available tasks with short description follows (with the assumption that the user is acknowledged with our paper). The exact list of expected parameters for each of the tasks can be simply read out from the [cli.py](https://github.com/crocs-muni/RSABias/blob/master/rsabias/cli.py). The tasks are presented in the order that leads to experiment replication.

How the tasks fit together:

```
datasets ──dist──► distributions ──group──► groups
   │                                          │
   └──split──► train ─────────build───────────┴──► model ──classify──► group of one key
               test ──filter──► evaluate ───────────────────────────► accuracy per group
```

`classify`, `evaluate` and `group` require the output directory (`-o`) to exist; `dist` and `convert` create it.

- `convert` -  Since we internally support multiple formats of the datasets, this task is capable of converting between the formats (csv -> json, enabling compression, etc.)
- `dist+plot` - Computes the exact distribution of the selected features on a whole dataset (a `dist.json` per source). Out of these features, labels for the dataset are constructed. This task also plots heat maps of the distributions, for sources whose observed feature values are few enough to draw. The two halves are also available separately as `dist` and `plot`.
- `group` - Performs the clustering task, using the distributions of the features as an input. Prints the groups as JSON and saves a dendrogram of the sources, showing how similar/distant the keys from the respective sources are. With a display, the dendrogram opens in a window, where a click sets the clustering threshold; without one (e.g. in Docker), the default threshold 0.1 is used, which reproduces [groups.json](rsabias/model/groups/groups.json).
- `marginal` - Computes marginal distributions of the features over the subspaces given by `-s`.
- `split` - Splits the dataset into training and test part. Out of each source, at least 10 000 keys are taken into the test set as default.
- `filter` - Prepares the test dataset. This task shuffles and merges the keys into a single artificially created dataset, that can be fed into the tasks below.
- `build` - Based on a list of transformations, groups obtained from clustering, and a dataset, this task constructs the classification tables. The CLI lets you select the naive or the complex Bayes classifier (`-m naive` or `-m complex`). The cross-feature classifier from the paper cannot be selected from the CLI. See Section 3 of the paper for more information.
- `evaluate` - Evaluates the classifier on a labelled dataset of keys and writes the model performance report as several json files. Every source in the input must be listed in the groups file (`-g`). Unlabelled datasets are not supported: the `-l` flag is accepted but ignored.
- `batch_gcd` - Classifies the GCD-factorable dataset (which only has a single prime per key) using existing classification tables. Before running it, replace the placeholder path `see/comment/above` in `Evaluator.init_batch_dictionary` ([classifier.py](https://github.com/crocs-muni/RSABias/blob/master/rsabias/core/classifier.py)) with the path to the dataset file. It needs a classification table built for single primes, which is not in the repository, and `ProbaTable` currently hardcodes the feature ranges and names of `5p_5q_blum_mod_roca.json` (`get_trans_ranges`, `get_feature_names`), so building one requires adapting them to `prime_wise.json`.
- `visualize` - To-be-called after the `evaluate` task. Takes the model performance results and prints them into a table. More info at [classification_table_template](https://github.com/crocs-muni/RSABias/tree/master/rsabias/classification_table_template).
- `classify` - Used to classify a single key. This is the main task that was already described above.

Each of the tasks can be invoked by calling

```bash
rsabias -a task_name task_arguments
```

The parameters of the tasks are:

- `-a` or `--action`: which action to perform, see above.
- `-t` or `--trans`: path to the file with transformations, see [model](https://github.com/crocs-muni/RSABias/tree/master/rsabias/model#transformations-description) for more information.
- `-i` or `--inp`: path to the input, whatever the task needs.
- `-o` or `--out`: path to the output, whatever the task produces.
- `-f` or `--format`: output format of the dataset for `convert`, `split` and `filter` (csv or json, default json).
- `-g` or `--groups`: path to the file with groups, see [model](https://github.com/crocs-muni/RSABias/tree/master/rsabias/model#groups-description) for more information.
- `-d` or `--decompress`: make `convert` and `split` write the output dataset uncompressed. Their output is gzip-compressed by default.
- `-s` or `--subspaces`: Subspaces for the `marginal` task.
- `-c` or `--classtable`: path to the **directory** with the classification tables (e.g. `rsabias/model`). `evaluate` loads `classification_table_naive` or `classification_table_complex` from it according to `-m`. `batch_gcd` always loads `classification_table_complex`. See [Model files](#model-files) for the file formats.
- `-m` or `--method`: which model to use, `complex` (default) or `naive` Bayes.
- `-l` or `--labels`: currently has no effect, see `evaluate` above.
- `-p` or `--prime_wise`: Whether single-prime model should be used, for gcd-factorized key classification.
- `-r` or `--remove_duplicities`: Whether duplicate keys should be deleted from a dataset.

To summarize, in order to fully replicate our experiments, one must do:

1. Download our datasets (see below).
2. Use our [feature vector](https://github.com/crocs-muni/RSABias/tree/master/rsabias/model#5p_5q_blum_mod_rocajson) to count the distribution of the features on the whole dataset. (`dist+plot` task). The output of this task can be found at [model/distributions](https://github.com/crocs-muni/RSABias/tree/master/rsabias/model/distributions) folder.
3. Split the key sources into multiple groups with the help of automated clustering. (`group` task). The output of this task can be found at [model/groups](https://github.com/crocs-muni/RSABias/tree/master/rsabias/model/groups) folder.
4. Split and filter the dataset into training and test part. (`split` and `filter` tasks)
5. Build the model. (`build` task)
6. Evaluate the performance of the model. (`evaluate` task)
7. Possibly classify the [batch-gcd dataset](https://github.com/crocs-muni/RSABias/tree/master/datasets/BatchGCD-primes-only). (`batch_gcd` task)
8. Possibly visualize the model performance. (`visualize` task)

We present a further description of the utilized data structures (groups, features, ...) in the [model folder](https://github.com/crocs-muni/RSABias/tree/master/rsabias/model).

#### Model files

Classification tables are stored as `.npz` files with plain arrays (probabilities per group, feature coordinates, group names), which load with any numpy version and without unpickling. The shipped model is [classification_table_complex.npz](rsabias/model/classification_table_complex.npz). `build` writes each table both as `.npz` and as a legacy `.pkl`; loading prefers the `.npz` and falls back to the `.pkl`.

The original model of the paper, [classification_table_complex.pkl](rsabias/model/classification_table_complex.pkl), is a pickle that only loads with the dependency versions of 2020. It was converted losslessly with [tools/convert_model.py](tools/convert_model.py) in the legacy environment; convert your own pickled tables from that time the same way. Tables built before the feature names `lsb` and `modularFingerprint` were corrected are relabelled automatically on load.

## Known limitations

- The model only knows sources collected until 2020 (see the top of this file).
- `classify` takes one key per run and opens it as text, so DER files fail; encrypted keys fail too; OpenSSH keys come out with `p` and `q` swapped. See [Supported key formats](#supported-key-formats).
- `classify`, `evaluate` and `group` fail with a traceback when the output directory does not exist.
- `evaluate` cannot classify an unlabelled set of keys: every dataset needs a group from the groups file, and `-l` is ignored.
- `group` prints the groups instead of saving them; `dist+plot` draws nothing when the feature space is too large to plot, which includes the model's transformation.
- `batch_gcd` needs a placeholder path replaced in the code and a single-prime model that is not in the repository.
- The cross-feature classifier from the paper cannot be selected from the CLI.

## Dataset

### Train & Test

We collected, analyzed, and published the largest dataset of RSA keys with a known origin from 70 libraries (43 open-source libraries, 5 black-box libraries, 3 HSMs, 19 smartcards). We both expanded the datasets from previous work and generated new keys from additional libraries for the sake of this study.

We are primarily interested in 2048-bit keys, which is the most commonly used key length for RSA. As in previous studies, we also generate shorter keys (512 and 1024 bits) to speed up the process, while verifying that the chosen biased features are not influenced by the key size. This makes the keys of different sizes interchangeable for the sake of our study. We assume that repeatedly running the key generation locally approximates the distributed behaviour of many instances of the same library.

The dataset of more than 160 million RSA keys can be accessed from [OwnCloud](https://owncloud.cesnet.cz/index.php/s/Ihhw3BKKzKTaxB9).

### GCD-factorable keys

We utilized the training data to classify a [Rapid-7](https://opendata.rapid7.com/sonar.ssl/) dataset of RSA keys factorable by a batch-GCD method. More on this dataset in [BatchGCD-primes-only](https://github.com/crocs-muni/RSABias/tree/master/datasets/BatchGCD-primes-only). Using this dataset, we classified more than 80 thousand weak RSA keys.

## Paper abstract

The paper can be found at [arXiv](https://arxiv.org/abs/2009.06700).

In 2016, Švenda et al. (USENIX 2016, The Million-key Question) reported that the implementation choices in cryptographic libraries allow for qualified guessing about the origin of public RSA keys.
We extend the technique to two new scenarios when not only public but also private keys are available for the origin attribution -- analysis of a source of GCD-factorable keys in IPv4-wide TLS scans and forensic investigation of an unknown source. We learn several representatives of the bias from the private keys to train a model on more than 150 million keys collected from 70 cryptographic libraries, hardware security modules and cryptographic smartcards. Our model not only doubles the number of distinguishable groups of libraries (compared to public keys from Švenda et al.) but also improves more than twice in accuracy w.r.t. random guessing when a single key is classified. For a forensic scenario where at least 10 keys from the same source are available, the correct origin library is correctly identified with average accuracy of 89\% compared to 4\% accuracy of a random guess. The technique was also used to identify libraries producing GCD-factorable TLS keys, showing that only three groups are the probable suspects.

## Project status & Contributing

This project is a result of a continuous effort of the [CRoCS](https://www.crocs.fi.muni.cz) laboratory that resulted into the following publications:

- [The Million-Key Question – Investigating the Origins of RSA Public Keys](https://crocs.fi.muni.cz/public/papers/usenix2016) - USENIX 2016, Best Paper Award,
- [The Return of Coppersmith's Attack: Practical Factorization of Widely Used RSA Moduli](https://crocs.fi.muni.cz/public/papers/rsa_ccs17) - ACM CCS 2017,
- [Measuring Popularity of Cryptographic Libraries in Internet-Wide Scans](https://crocs.fi.muni.cz/public/papers/acsac2017) - ACSAC 2017,
- [Biased RSA private keys: Origin attribution of GCD-factorable keys](https://crocs.fi.muni.cz/public/papers/privrsa_esorics20) - ESORICS 2020.

If you would like to contribute, feel free to contact [Adam Janovsky](https://github.com/adamjanovsky) or open an issue.

## Authors

[Adam Janovsky](https://github.com/adamjanovsky), [Matus Nemec](https://github.com/matusn), [Petr Svenda](https://github.com/petrs), [Peter Sekan](https://github.com/psekan) and [Vashek Matyas](https://matyas.cz); all from [Masaryk University](https://www.muni.cz), [Faculty of Informatics](https://www.fi.muni.cz), [Center for Research on Cryptography and Security](https://crocs.fi.muni.cz) in Czech Republic.

## License

This tool is available under [MIT License](https://github.com/matusn/RSABias/blob/master/LICENSE).
