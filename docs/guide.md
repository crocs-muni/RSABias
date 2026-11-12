# RSABias: a practical guide for testing RSA key generation

This guide shows how to use RSABias to examine RSA keys produced by your own key generation: which known library or device they resemble, whether two sets of keys differ, and whether your keys show anomalies compared with the datasets the tool was built from.

This guide is a set of walkthroughs; the reference for every command and option is the [README](../README.md).

## 1. What RSABias does

RSABias accompanies the paper [Biased RSA private keys: Origin attribution of GCD-factorable keys](https://arxiv.org/abs/2009.06700) (ESORICS 2020). Implementations generate RSA primes in different ways, and those choices leave statistical traces in the keys. The tool measures five of them on every private key:

| Feature | What it measures |
|---|---|
| msb5 of p, msb5 of q | the 5 most significant bits of each prime, read separately |
| Blum | whether both primes are 3 mod 4 |
| modular fingerprint | the smallest small-prime bound up to which p-1 and q-1 have no small odd divisors: 1 (none), 5, 251 or 17863 |
| ROCA | whether the modulus has the structure of the Infineon ROCA vulnerability (CVE-2017-15361) |

From more than 150 million keys of 70 sources (libraries, smartcards, HSMs) collected until 2020, the sources were clustered into **26 groups** of sources whose keys are indistinguishable by these features. A shipped model (a Bayes classifier) then estimates which group a key most likely comes from.

What follows from this:

- **The answer is a group, not a library version.** Group 24, for example, holds OpenSSL 0.9.7 to 1.1.0e, including FIPS builds.
- **The tool needs private keys.** p and q are required; public keys and certificates cannot be analysed.
- **One key says little.** For a single key the model is right in 47.7% of cases on average (a random guess: 4%). With 10 keys from one source, it is right in about 89%.
- **The model only knows sources until 2020.** Keys from other or newer implementations are still assigned the closest group. For example, OpenSSL 3 keys are not recognised as OpenSSL: their modular fingerprint is mostly 1, while the OpenSSL versions in the model always give 17863. Comparing distributions (walkthrough 3) shows such differences directly.
- **Only these five features are compared.** A change in key generation that leaves them unchanged is not visible to the tool.

## 2. Handling keys

Everything in this guide runs on your machines; private keys never need to leave them. What you may share with us when reporting results or problems:

| Shareable | Not shareable |
|---|---|
| `dist.json` and `meta.json` from `dist` (counts of feature values) | key files, `keys.json` dataset files |
| `evaluate` outputs (`n_*_results.json`, `confusion_matrix.*`) | anything containing p, q, d or n |
| `key_classification.json` reports (scores per group, no key material) | |

Work on copies of the keys, never on the originals.

## 3. Setup

Clone the repository and run every command in this guide from its root:

```bash
git clone https://github.com/crocs-muni/RSABias.git
cd RSABias
mkdir -p work        # keys and outputs go here; git ignores this directory
```

**Docker (recommended).** Build the image once, then define a shell function so that the commands below work unchanged. The function mounts the repository root, so all paths must lie inside it (hence `work/`):

```bash
docker build -t rsabias .
rsabias() { docker run --rm --user "$(id -u):$(id -g)" -v "$PWD:/work" rsabias "$@"; }
rsabias-python() { docker run --rm --user "$(id -u):$(id -g)" -v "$PWD:/work" --entrypoint python rsabias "$@"; }
```

**Local install** (Python 3.11 or newer):

```bash
python3 -m venv venv && source venv/bin/activate && pip install .
alias rsabias-python=python
```

Check the setup with `rsabias -h`.

**Output directories must exist** before `classify`, `evaluate` and `group` run; only `dist` creates its own. Each walkthrough creates them with `mkdir -p`.

The model files used below are in `rsabias/model/`:

```bash
TRANS=rsabias/model/transformations/5p_5q_blum_mod_roca.json
GROUPS_JSON=rsabias/model/groups/groups.json
MODEL=rsabias/model
```

## 4. Walkthrough 1: classify one key

Generate a test key, classify it and read the result:

```bash
mkdir -p work/w1
openssl genrsa -out work/w1/key.pem 2048
rsabias -a classify -i work/w1/key.pem -o work/w1
```

The output names the most likely group, its precision ("if a key is marked as coming from group X, the model is right in Y% of cases"), its recall, and the sources in the group. `work/w1/key_classification.json` holds the same for all 26 groups, ordered by score. A score of 0 means the group has never produced this combination of features: such groups are ruled out, which for a single key is often the most solid result.

Key formats: the file must hold one unencrypted RSA private key in PEM (PKCS#1 or PKCS#8). Convert other formats first:

```bash
openssl rsa -inform DER -in key.der -out key.pem              # DER
openssl rsa -in encrypted.pem -out key.pem                    # remove a passphrase
cp id_rsa key.pem && ssh-keygen -p -m PEM -N "" -f key.pem    # OpenSSH format
```

Do not classify OpenSSH-formatted keys directly: they are read with p and q swapped, which changes the result without an error.

## 5. Walkthrough 2: which known source do my keys resemble?

With several keys **from one source**, combine their results: the scores of each group are multiplied over all keys, as the model's own evaluation does. Classify each key into its own directory (every run overwrites `key_classification.json`), then summarise:

```bash
# keys in work/mykeys/*.pem
for f in work/mykeys/*.pem; do
    n=$(basename "$f" .pem); mkdir -p "work/w2/$n"
    rsabias -a classify -i "$f" -o "work/w2/$n" > /dev/null
done
rsabias-python tools/summarize_classification.py work/w2/*/key_classification.json
```

The summary lists the top group of every key, how often each group came first, and the combined ranking with the sources of the top three groups. Groups that any key rules out are excluded.

Reading it:

- Agreement between the per-key top groups and the combined ranking supports the verdict. A combined winner that few keys picked individually means the keys fit no group well.
- If **every** group is ruled out, the keys either come from several sources or from a source the model does not know.
- Only combine keys from one source. Keys from different sources rule out each other's groups.

Each `classify` run takes a few seconds, so this suits up to a few hundred keys. For large sets, use walkthrough 3.

## 6. Walkthrough 3: compare key sets

This is the main tool for comparing two sets of your keys (for example two versions of your library), or your keys with the sources the model was built from. It works with any number of keys: `dist` processes about 1 million keys in 5 minutes.

### Step 1: pack the keys into a dataset

`dist` does not read loose key files but datasets. A dataset is a directory with the keys of one source in a key file (here a concatenation of PEM keys) and a `meta.json` that describes the source (category, name, version, key length) and lists the key files with their number of keys:

```json
{
  "type": "reference",
  "files": [{"name": "keys.pem", "records": 1000000}],
  "details": {"category": "Library", "name": "MyLib", "version": "1.0", "bitlen": 2048,
              "format": "pem", "compressed": false, "fips_mode": false}
}
```

`dist` searches its input directory for all `meta.json` files, so one input directory can hold several sources. Other key file formats (JSON lines, CSV) are described in [datasets/baby_dataset/readme.md](../datasets/baby_dataset/readme.md).

`tools/pem2dataset.py` creates a dataset from PEM files. Arguments: output directory, source name, version, key files:

```bash
rsabias-python tools/pem2dataset.py work/datasets/v1/mylib-v1 MyLib 1.0 work/keys-v1/*.pem
rsabias-python tools/pem2dataset.py work/datasets/v2/mylib-v2 MyLib 2.0 work/keys-v2/*.pem
```

It skips files it cannot read and keys of a different length than the first, with a warning.

Or create it with the following script, which also works when there are too many keys for one `*.pem` argument list. Set the source details and the key length in `meta.json` to your keys:

```bash
src=work/keys-v1; out=work/datasets/v1/mylib-v1
mkdir -p "$out"
find "$src" -name '*.pem' -exec cat {} + > "$out/keys.pem"
n=$(grep -c -- '-----BEGIN .*PRIVATE KEY-----' "$out/keys.pem")
cat > "$out/meta.json" <<EOF
{
  "type": "reference",
  "files": [{"name": "keys.pem", "records": $n}],
  "details": {"category": "Library", "name": "MyLib", "version": "1.0", "bitlen": 2048,
              "format": "pem", "compressed": false, "fips_mode": false}
}
EOF
```

### Step 2: compute the distributions

```bash
rsabias -a dist -i work/datasets/v1 -o work/dist-v1 -t $TRANS
rsabias -a dist -i work/datasets/v2 -o work/dist-v2 -t $TRANS
```

Each source gets a `dist.json` with the count of every combination of feature values, under `<output>/<category>/<name>/<version>/<bitlen>/`.

### Step 3: compare

```bash
rsabias-python tools/compare_dist.py work/dist-v1 work/dist-v2
```

For each feature, it prints the share of every value in both sets, the difference in percentage points, and a chi-square test; then the total variation distance over all features together, the combinations that differ most, and combinations seen in only one set. An excerpt, comparing the model's 100,000 OpenSSL 1.1.0e keys with 1,000 OpenSSL 3.5.3 keys:

```
modular fingerprint (chi-square p = 0)
     value      A %      B %    B-A pp
         1     0.00    84.80    +84.80
         5     0.00    13.00    +13.00
       251     0.00     1.60     +1.60
     17863   100.00     0.60    -99.40
```

Reading it: with many keys, even negligible differences are statistically significant, so judge them by the size in percentage points. For scale, OpenSSL 1.1.0e keys of 1024 and 2048 bits, which come from the same implementation, differ by a total variation distance of 1.2 pp.

### Comparing with the sources of the model

The distributions of all 184 sources (source, version and key length) the model was built from are in `rsabias/model/distributions/`. Either side of `compare_dist.py` can be a single `dist.json` or a directory, whose distributions are summed:

```bash
# against one known source
rsabias-python tools/compare_dist.py rsabias/model/distributions/Library/OpenSSL/1.1.0e/2048/dist.json work/dist-v1
# against all versions of a library
rsabias-python tools/compare_dist.py rsabias/model/distributions/Library/OpenSSL work/dist-v1
```

To see which known sources are closest overall, cluster your distribution together with all of them. Your source either joins an existing group or forms its own:

```bash
mkdir -p work/cluster/in work/cluster/out
cp -r rsabias/model/distributions/* work/cluster/in/
cp -r work/dist-v1/* work/cluster/in/
rsabias -a group -i work/cluster/in -o work/cluster/out -t $TRANS > work/cluster/out/groups.txt
```

`groups.txt` lists the resulting groups as JSON; find your source in it to see which known sources share its group. The dendrogram PDF in `work/cluster/out` shows how far each source is from the others.

### Limitations

- `compare_dist.py` is text only. A visual comparison of two distributions is on the potential roadmap of [scrutiny-viz](https://github.com/crocs-muni/scrutiny-viz).
- `group` prints its groups instead of saving them, and its dendrogram file has a long, generated name.
- `dist+plot` draws nothing for the model's transformation (its feature space is too large to plot).

## 7. Advanced usage

The walkthroughs above look at your keys: `classify` scores them against the model, and `dist`, `compare_dist.py` and `group` compare the feature distributions of your key set directly with other key sets. The remaining tasks take a different approach and are documented in the [README](../README.md#advanced-usage) and the code rather than here.

**Evaluating the model.** `evaluate` does not describe keys; it measures the model. It takes keys whose true group is known (the ground truth, `group` in each `meta.json`), classifies them alone and in batches of 10 to 100 keys, and counts how often the model names the true group. The paper used it to measure the accuracy of the model on the test part of the dataset. Run on your keys labelled with the group you expect, it shows how well the classification built into the model recognises them as that group. This differs from comparing with `dist`: it does not compare your keys with other keys, only with the model's 26 groups, so a difference in the features shows only where it changes the classification. It is also slow (about 5 hours per million keys). See [Evaluating keys against a group](../README.md#evaluating-keys-against-a-group) in the README.

**Building a model.** The model was built by computing the distributions of all sources (`dist`), clustering them into groups (`group`), and building the classification tables from a training part of the dataset (`split`, `filter`, `build`), see the pipeline in the README. Adding your own implementation as a new group means repeating these steps with your keys added; contact us if you need it.

## 8. Interpreting results

- **Report a group, not a library.** "Most consistent with group 24 (OpenSSL 0.9.7 to 1.1.0e); the model is right 90% of the time when it predicts group 24; 13 groups are ruled out" is the right register.
- **Precision varies a lot between groups** (`rsabias/model/precision_recall.json`). Some groups are almost never predicted correctly; quote the precision of the top group.
- **Ruled-out groups are the most reliable single-key result.**
- **Unknown sources are assigned anyway.** If the summary of walkthrough 2 rules out every group, or `compare_dist.py` shows large differences from every known source, your keys come from a source the model does not know. That is a finding in itself.
- **Differences between two of your sets** in walkthrough 3 show a change in how the primes are generated. Small differences in percentage points can be sampling noise even if the p-value is tiny.

## 9. Troubleshooting

| Problem | Cause and fix |
|---|---|
| `FileNotFoundError` at the end of `classify`, `evaluate` or `group` (`classify` prints its result first, then fails writing the report) | The output directory does not exist: `mkdir -p` it first |
| `Permission denied` when writing outputs in Docker | The output directory was created by Docker as root. Remove it, create it yourself, and run with `--user "$(id -u):$(id -g)"` (the `rsabias` function above does) |
| Docker cannot find a file | Paths must lie inside the directory you run from (it is mounted as `/work`); use relative paths |
| `Failed to read key-data from file, 'utf-8' codec can't decode` | DER key, convert it to PEM |
| `The private key is encrypted: a passphrase is required` | Remove the passphrase from a copy |
| `TypeError ... 'NoneType'` in `count_records` during `evaluate` | `records` is missing in `meta.json` |
| `KeyError` in `evaluate` | `details.group` is missing in `meta.json`, or names a group that is not in `$GROUPS_JSON` |

When something fails, send us: the command, the full error output, the `meta.json` files involved, `git rev-parse --short HEAD` of your checkout, and whether you used Docker or a local install. Never send keys.
