---
name: prepare-rsa-keys
description: Prepare a user's RSA keys for classification with rsabias. Use when someone wants to classify or analyse their own RSA keys and has them in any form (PEM, DER, PKCS#8, encrypted, OpenSSH, PuTTY, PKCS#12, JWK, raw p/q numbers, JSON lines). Walks through what they have, converts it into one unencrypted PKCS#1 PEM per key without changing the p/q order, and reports what could not be used.
---

# Prepare RSA keys for rsabias

`rsabias -a classify` takes **one unencrypted RSA private key per file, in PEM**. Your job is to work out what the user has, convert it into that format without losing information the classifier relies on, and say clearly what could not be converted and why. Classification itself is a separate step (see the `rsa-key-analyst` agent or the README).

## Ground rules

- **Private keys are secrets.** Never print, quote or paste key material (PEM bodies, p, q, d, n in full) into the conversation, commit it, or send it to any external service. Refer to keys by file name or by the `n_sha256` fingerprint from the manifest.
- **Work on copies.** Never modify the user's original files. Write output to a fresh directory outside the repository, such as a temp or scratch directory. The script creates it with mode 0700 and the key files with 0600.
- **Passphrases never go on the command line or in chat.** Ask the user to put the passphrase in a file only they can read, from their own terminal (for example `read -s P; printf '%s\n' "$P" > ~/.keypass; chmod 600 ~/.keypass` in bash/zsh), and pass `--passphrase-file ~/.keypass`. Don't read that file yourself, and remind the user to delete it afterwards. Environment variables don't persist between Claude's shell calls, so `--passphrase-env` only helps when the user runs the command themselves.
- When done, remind the user where the converted copies are and that they should delete them after classification.

## Why the p/q order matters

The classifier's features (`rsabias/model/transformations/5p_5q_blum_mod_roca.json`) read the most significant bits of `p` and of `q` separately. Which prime a library stores first is part of its fingerprint, so a conversion that swaps them can change the result. The same key gives group 24 in its original order and group 6 when swapped. The script keeps the order exactly as stored in the source:

- PKCS#1, PKCS#8, DER and JWK store p and q explicitly, and the stored order is kept.
- **OpenSSH**: pycryptodome (used by `rsabias` itself) swaps p and q when importing OpenSSH keys. The script parses unencrypted OpenSSH keys itself to avoid that. For encrypted OpenSSH keys, have the user convert a copy with `ssh-keygen -p -m PEM -f COPY`, which keeps the order.
- **Raw primes** (e.g. from GCD factorisation): the original order is usually unknown. Convert both orders (second run with `--swap-pq` into a different directory), classify both, and report whether they agree.

## Step 1: find out what the user has

Ask only what you can't see from the files. Most of it you can tell by looking at the first line of a file (`head -c 64`, which shows the PEM header but not key material) or with `file`:

1. **Where are the keys and how many?** A single file, a directory, or a bundle with several keys in one file?
2. **What form are they in?** Use the table below. Public keys and certificates cannot be classified, because the model needs the private primes. Say so early.
3. **Are they protected by a passphrase?**
4. **Do they come from the same source?** Several keys from one device or library can be combined into a much stronger verdict than a single key. Keys from different sources must be classified separately. Find out which keys belong together, because this decides how the results are combined.
5. **What is already known about where they came from?** For example the software or device, roughly when they were generated, or whether they were imported from elsewhere. This helps with interpreting the result later, not with converting.

| What the user has | Recognise it by | What to do |
|---|---|---|
| PKCS#1 PEM | `-----BEGIN RSA PRIVATE KEY-----` | Already accepted by the tool, but running it through the script is harmless and gives a manifest |
| PKCS#8 PEM | `-----BEGIN PRIVATE KEY-----` | Script |
| Encrypted PEM | `-----BEGIN ENCRYPTED PRIVATE KEY-----` or `Proc-Type: 4,ENCRYPTED` | Script with `--passphrase-file` |
| DER (binary) | `file` says data/DER, often `.der`/`.key` | Script (with `--passphrase-file` if encrypted) |
| OpenSSH | `-----BEGIN OPENSSH PRIVATE KEY-----` | Script if unencrypted. If encrypted, the user runs `ssh-keygen -p -m PEM -f COPY` on a copy, then the script |
| PuTTY | `PuTTY-User-Key-File-` | User runs `puttygen KEY.ppk -O private-openssh -o KEY.pem` on a copy, then the script (not tested here) |
| PKCS#12 | `.p12` / `.pfx` | User runs `openssl pkcs12 -in FILE -nocerts -nodes -out KEY.pem`, then the script |
| JWK / JWKS | JSON with `"kty": "RSA"` | Script (public-only entries are skipped) |
| Numbers | JSON object or JSON lines with `p`, `q` and optionally `n`, `e`, `d`, as ints, hex or decimal strings (the rsabias dataset format is hex JSON lines) | Script. Pass `--default-e` if `e` is missing. Without `n`, check the base it picked (warning in the manifest) and force it with `--int-base` if it's wrong |
| Plain text dumps (e.g. `openssl rsa -text` output) | `prime1:` / `prime2:` hex blocks | Write them into a JSON object `{"n": ..., "e": ..., "p": prime1, "q": prime2}` with hex strings yourself, without echoing the values, then the script. `prime1` is p |
| Public key, certificate, CSR, `ssh-rsa AAAA...` line | `PUBLIC KEY`, `CERTIFICATE`, one-line `ssh-rsa` | Cannot be classified. Explain why |
| Hardware key (smartcard, HSM, TPM) | — | Usually not exportable. The tool can only analyse keys whose private part is available |

## Step 2: convert

The script needs Python 3.7+ and pycryptodome, which are rsabias's own dependencies. Run it with `-I` so it doesn't import anything from the directory holding the keys.

From an environment where rsabias works:

```bash
python -I .claude/skills/prepare-rsa-keys/scripts/normalize_keys.py INPUT [INPUT ...] -o OUT_DIR \
    [--passphrase-file PASSFILE] [--default-e 65537] [--int-base 16] [--swap-pq]
```

Or with the rsabias Docker image (`docker build -t rsabias .`). Mount the keys read-only, run as your own user, and create the output directory first: Docker creates a missing one owned by root and the script then cannot write into it.

```bash
mkdir -p /path/to/out
docker run --rm --entrypoint python --user "$(id -u):$(id -g)" \
    -v "$PWD/.claude/skills/prepare-rsa-keys/scripts:/s:ro" \
    -v "/path/to/keys:/in:ro" -v "/path/to/out:/out" \
    [-v "$HOME/.keypass:/pass:ro"] rsabias -I /s/normalize_keys.py /in -o /out [--passphrase-file /pass]
```

Inputs can be files or directories (searched recursively). Files with several PEM blocks, JWKS sets and JSON lines produce one key per entry.

What the script does:
- Writes `0000_<fingerprint>.pem`, `0001_...` (PKCS#1 PEM, one key each) and `manifest.json` to `OUT_DIR`.
- Checks every key: `n = p*q`, p and q are prime, `e*d = 1 mod lcm(p-1, q-1)`. It computes `n` or `d` if they are missing and records that.
- Drops duplicates by modulus, so the same key in two encodings counts once.
- Re-reads each written PEM and aborts if the p/q order changed.
- Prints only metadata: file, source, format, bit length, `p>q`, warnings, skip reasons.

## Step 3: review and report back

Read the summary output and `manifest.json` (it holds no key material) and tell the user:
- how many keys were converted, and where they are;
- every skipped input with its reason in plain words, and the next step if there is one (e.g. "this is a certificate; I need the matching private key");
- duplicates, so they know two files were the same key;
- warnings that affect interpretation: a computed `n`/`d`, an assumed `e`, a guessed number base, a swapped order;
- whether the keys in the batch look consistent. A mix of `p>q` True and False among keys that are supposed to come from the same source is worth pointing out, since a library usually stores the primes in the same order every time (OpenSSL, for example, always stores the larger prime as p).

Then hand over to classification: one `rsabias -a classify -i KEY.pem -o REPORT_DIR` per key, with a separate report directory per key, since every run overwrites `key_classification.json`. The `rsa-key-analyst` agent covers running and interpreting this, including combining keys from the same source.
