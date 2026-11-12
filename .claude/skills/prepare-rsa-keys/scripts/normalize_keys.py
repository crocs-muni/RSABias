#!/usr/bin/env python3
"""
Convert RSA private keys from common encodings into the input expected by
`rsabias -a classify`: one unencrypted PKCS#1 PEM key per file.

The order of the primes p and q is preserved exactly as stored in the source,
because the classifier's features treat p and q differently and the order is
itself a trace of the generating library.

Never prints key material. Writes a manifest.json with per-key metadata
(source, input format, size, fingerprint of n, p > q, warnings).

Requires only the standard library and pycryptodome (a dependency of rsabias),
and runs on Python 3.7+, so it works inside the rsabias Docker image too.

Usage:
    python -I normalize_keys.py INPUT [INPUT ...] -o OUT_DIR
        [--passphrase-file FILE | --passphrase-env VAR] [--default-e 65537] [--int-base auto|10|16]
        [--swap-pq]
"""

import argparse
import base64
import hashlib
import json
import os
import re
import struct
import sys

try:
    from Crypto.PublicKey import RSA
    from Crypto.Util.number import isPrime
except ImportError:
    sys.exit('pycryptodome is required (pip install pycryptodome), '
             'or run this script inside the rsabias Docker image.')

PEM_BLOCK = re.compile(
    rb'-----BEGIN ([A-Z0-9 ]+)-----\r?\n(.*?)-----END \1-----', re.S)


class SkipKey(Exception):
    """The input is understood but cannot be turned into a classifiable key."""


# ---------------------------------------------------------------- arithmetic

def inverse(a, m):
    """Modular inverse (pow(a, -1, m) needs Python 3.8)."""
    old_r, r, old_s, s = a % m, m, 1, 0
    while r:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_s, s = s, old_s - q * s
    if old_r != 1:
        raise SkipKey('e is not invertible modulo lcm(p-1, q-1)')
    return old_s % m


def lcm(a, b):
    from math import gcd
    return a // gcd(a, b) * b


def fingerprint(n):
    """SHA-256 of the big-endian modulus, as hex. Identifies the key, not the file."""
    return hashlib.sha256(n.to_bytes((n.bit_length() + 7) // 8, 'big')).hexdigest()


def build(n, e, d, p, q, warnings):
    """Validate the components and return them with p, q in the given order."""
    if n is None:
        n = p * q
        warnings.append('n was missing and was computed as p*q')
    if n != p * q:
        raise SkipKey('n != p*q')
    if not (isPrime(p) and isPrime(q)):
        raise SkipKey('p or q is not prime')
    if e is None:
        raise SkipKey('public exponent e is missing (pass --default-e)')
    lam = lcm(p - 1, q - 1)
    if d is None:
        d = inverse(e, lam)
        warnings.append('d was missing and was computed from e')
    elif (e * d) % lam != 1:
        raise SkipKey('e*d != 1 mod lcm(p-1, q-1)')
    return n, e, d, p, q


# ---------------------------------------------------------------- parsers

def parse_pycryptodome(data, passphrase):
    """PKCS#1 / PKCS#8 (plain or encrypted), PEM or DER. Keeps the stored p, q order."""
    # pycryptodome rejects unencrypted PKCS#8 when a passphrase is given, so
    # try without one first.
    key = None
    for pw in [None] + ([passphrase] if passphrase is not None else []):
        try:
            key = RSA.import_key(data, passphrase=pw)
            break
        except (ValueError, IndexError, TypeError) as ex:
            msg = str(ex)
    if key is None:
        hint = '' if passphrase is not None else ' (if the key has a passphrase, pass --passphrase-file)'
        raise SkipKey('cannot parse: {}{}'.format(msg, hint))
    if not key.has_private():
        raise SkipKey('public key only; the classifier needs the private primes p and q')
    return int(key.n), int(key.e), int(key.d), int(key.p), int(key.q)


def _ssh_read(buf, off):
    (length,) = struct.unpack('>I', buf[off:off + 4])
    off += 4
    return buf[off:off + length], off + length


def parse_openssh(block_body):
    """
    Unencrypted OpenSSH private key. Parsed by hand because pycryptodome's
    importer swaps p and q (it builds the key from (n, e, d, q, p, iqmp)).
    """
    raw = base64.b64decode(b''.join(block_body.split()))
    magic = b'openssh-key-v1\x00'
    if not raw.startswith(magic):
        raise SkipKey('not an openssh-key-v1 blob')
    off = len(magic)
    cipher, off = _ssh_read(raw, off)
    kdf, off = _ssh_read(raw, off)
    _, off = _ssh_read(raw, off)  # kdf options
    if cipher != b'none' or kdf != b'none':
        raise SkipKey('encrypted OpenSSH key; convert a copy with '
                      '`ssh-keygen -p -m PEM -f COPY` (keeps the p, q order) and rerun')
    (nkeys,) = struct.unpack('>I', raw[off:off + 4])
    off += 4
    for _ in range(nkeys):
        _, off = _ssh_read(raw, off)  # public keys
    priv, off = _ssh_read(raw, off)
    off = 8  # two check integers
    keytype, off = _ssh_read(priv, off)
    if keytype != b'ssh-rsa':
        raise SkipKey('OpenSSH key of type {} is not RSA'.format(keytype.decode(errors='replace')))
    vals = []
    for _ in range(6):  # n, e, d, iqmp, p, q
        v, off = _ssh_read(priv, off)
        vals.append(int.from_bytes(v, 'big'))
    n, e, d, _iqmp, p, q = vals
    return n, e, d, p, q


def b64url_int(s):
    s = s + '=' * (-len(s) % 4)
    return int.from_bytes(base64.urlsafe_b64decode(s), 'big')


def parse_number(v, base):
    """Parse an int, a 0x-prefixed hex string, or a string in the given base."""
    if isinstance(v, int):
        return v
    s = str(v).strip().lower()
    if s.startswith('0x'):
        return int(s, 16)
    return int(s, base)


def parse_json_obj(obj, int_base, default_e):
    """JWK (kty=RSA) or a dict with n/e/d/p/q as ints, hex or decimal strings."""
    if obj.get('kty') == 'RSA':
        if 'p' not in obj:
            raise SkipKey('public JWK only; the classifier needs p and q')
        comps = {k: b64url_int(obj[k]) for k in ('n', 'e', 'd', 'p', 'q') if k in obj}
        return comps.get('n'), comps.get('e'), comps.get('d'), comps['p'], comps['q'], 'jwk', None
    if 'p' not in obj or 'q' not in obj:
        raise SkipKey('JSON object without both p and q')
    # Without an explicit base, take the first base in which p and q are prime
    # and n (when present) equals p*q. Hex first: the rsabias datasets use it.
    bases = [int_base] if int_base in (10, 16) else [16, 10]
    for b in bases:
        try:
            p, q = parse_number(obj['p'], b), parse_number(obj['q'], b)
            n = parse_number(obj['n'], b) if 'n' in obj else None
        except ValueError:
            continue
        if (n is not None and n != p * q) or not (isPrime(p) and isPrime(q)):
            continue
        e = parse_number(obj['e'], b) if 'e' in obj else default_e
        d = parse_number(obj['d'], b) if 'd' in obj else None
        note = None
        if int_base not in (10, 16):
            note = 'numbers interpreted as base {}{}'.format(
                b, '' if n is not None else ' (no n to confirm it; pass --int-base if wrong)')
        if 'e' not in obj:
            note = (note + '; ' if note else '') + 'e missing, used --default-e {}'.format(default_e)
        return n, e, d, p, q, 'json-{}'.format(b), note
    raise SkipKey('could not interpret p, q, n consistently in base 10 or 16')


# ---------------------------------------------------------------- driver

def iter_inputs(paths):
    for path in paths:
        if os.path.isdir(path):
            for root, dirs, files in os.walk(path):
                dirs.sort()
                for f in sorted(files):
                    yield os.path.join(root, f)
        else:
            yield path


def extract(path, data, passphrase, int_base, default_e):
    """Yield (label, fmt, components-or-exception, note) for every key found in a file."""
    if data.startswith(b'PuTTY-User-Key-File'):
        yield path, 'ppk', SkipKey('PuTTY key; convert a copy with '
                                   '`puttygen KEY.ppk -O private-openssh -o KEY.pem` and rerun'), None
        return
    blocks = list(PEM_BLOCK.finditer(data))
    if blocks:
        for i, m in enumerate(blocks):
            label = '{}#{}'.format(path, i) if len(blocks) > 1 else path
            kind = m.group(1).decode()
            try:
                if kind == 'OPENSSH PRIVATE KEY':
                    yield label, 'openssh', parse_openssh(m.group(2)), 'p, q read in OpenSSH order'
                elif kind in ('RSA PRIVATE KEY', 'PRIVATE KEY', 'ENCRYPTED PRIVATE KEY'):
                    fmt = {'RSA PRIVATE KEY': 'pkcs1-pem', 'PRIVATE KEY': 'pkcs8-pem',
                           'ENCRYPTED PRIVATE KEY': 'pkcs8-pem-encrypted'}[kind]
                    if kind == 'RSA PRIVATE KEY' and b'ENCRYPTED' in m.group(2)[:200]:
                        fmt = 'pkcs1-pem-encrypted'
                    yield label, fmt, parse_pycryptodome(m.group(0), passphrase), None
                elif kind in ('PUBLIC KEY', 'RSA PUBLIC KEY', 'CERTIFICATE', 'CERTIFICATE REQUEST'):
                    yield label, kind.lower(), SkipKey('{} only; the classifier needs the private key'.format(kind.lower())), None
                else:
                    yield label, kind.lower(), SkipKey('unsupported PEM block "{}" (not an RSA private key)'.format(kind)), None
            except SkipKey as ex:
                yield label, kind.lower(), ex, None
        return
    stripped = data.strip()
    if stripped[:1] in (b'{', b'['):
        try:
            text = stripped.decode('utf-8')
            try:
                docs = [json.loads(text)]
            except ValueError:  # JSON lines, e.g. the rsabias dataset format
                docs = [json.loads(line) for line in text.splitlines() if line.strip()]
        except ValueError as ex:
            yield path, 'json', SkipKey('invalid JSON: {}'.format(ex)), None
            return
        objs = []
        for doc in docs:
            if isinstance(doc, dict) and isinstance(doc.get('keys'), list):
                objs.extend(doc['keys'])
            elif isinstance(doc, list):
                objs.extend(doc)
            else:
                objs.append(doc)
        for i, obj in enumerate(objs):
            label = '{}#{}'.format(path, i) if len(objs) > 1 else path
            try:
                if not isinstance(obj, dict):
                    raise SkipKey('JSON item is not an object')
                n, e, d, p, q, fmt, note = parse_json_obj(obj, int_base, default_e)
                yield label, fmt, (n, e, d, p, q), note
            except SkipKey as ex:
                yield label, 'json', ex, None
        return
    # Binary: DER (PKCS#1 or PKCS#8, possibly encrypted). PKCS#12 is also DER
    # but pycryptodome cannot read it.
    try:
        yield path, 'der', parse_pycryptodome(data, passphrase), None
    except SkipKey as ex:
        if path.lower().endswith(('.p12', '.pfx')):
            ex = SkipKey('PKCS#12 bundle; extract the key from a copy with '
                         '`openssl pkcs12 -in FILE -nocerts -nodes -out KEY.pem` and rerun')
        yield path, 'binary', ex, None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('inputs', nargs='+', help='key files or directories (searched recursively)')
    ap.add_argument('-o', '--out', required=True, help='output directory (created with mode 0700)')
    ap.add_argument('--passphrase-file', help='file whose first line is the passphrase')
    ap.add_argument('--passphrase-env', help='name of an environment variable holding the passphrase')
    ap.add_argument('--default-e', type=int, default=None,
                    help='public exponent to assume for raw p/q inputs that lack e')
    ap.add_argument('--int-base', choices=['auto', '10', '16'], default='auto',
                    help='base of number strings in JSON input (default: detect via n = p*q)')
    ap.add_argument('--swap-pq', action='store_true',
                    help='swap p and q in every output key (to test the other order when it is unknown)')
    args = ap.parse_args()

    passphrase = None
    if args.passphrase_file:
        with open(args.passphrase_file, 'r') as fh:
            passphrase = fh.readline().rstrip('\r\n')
    elif args.passphrase_env:
        passphrase = os.environ.get(args.passphrase_env)
        if passphrase is None:
            sys.exit('environment variable {} is not set'.format(args.passphrase_env))
    int_base = int(args.int_base) if args.int_base != 'auto' else None

    os.makedirs(args.out, mode=0o700, exist_ok=True)
    os.umask(0o077)

    manifest = {'converted': [], 'skipped': [], 'duplicates': []}
    seen = {}
    for path in iter_inputs(args.inputs):
        try:
            with open(path, 'rb') as fh:
                data = fh.read()
        except OSError as ex:
            manifest['skipped'].append({'source': path, 'reason': str(ex)})
            continue
        for label, fmt, comps, note in extract(path, data, passphrase, int_base, args.default_e):
            if isinstance(comps, Exception):
                manifest['skipped'].append({'source': label, 'format': fmt, 'reason': str(comps)})
                continue
            warnings = [note] if note else []
            try:
                n, e, d, p, q = build(*comps, warnings=warnings)
            except SkipKey as ex:
                manifest['skipped'].append({'source': label, 'format': fmt, 'reason': str(ex)})
                continue
            if args.swap_pq:
                p, q = q, p
                warnings.append('p and q swapped by --swap-pq')
            fp = fingerprint(n)
            if fp in seen:
                manifest['duplicates'].append({'source': label, 'same_as': seen[fp]})
                continue
            out_name = '{:04d}_{}.pem'.format(len(manifest['converted']), fp[:16])
            pem = RSA.construct((n, e, d, p, q), consistency_check=True).export_key('PEM')
            check = RSA.import_key(pem)
            if (int(check.p), int(check.q)) != (p, q):
                raise RuntimeError('p/q order changed on export; refusing to continue')
            with open(os.path.join(args.out, out_name), 'wb') as fh:
                fh.write(pem + b'\n')
            seen[fp] = label
            manifest['converted'].append({
                'file': out_name, 'source': label, 'format': fmt, 'bits': n.bit_length(),
                'n_sha256': fp, 'e': e, 'p_gt_q': p > q, 'warnings': warnings})

    with open(os.path.join(args.out, 'manifest.json'), 'w') as fh:
        json.dump(manifest, fh, indent=2)

    print('converted: {}  skipped: {}  duplicates: {}'.format(
        len(manifest['converted']), len(manifest['skipped']), len(manifest['duplicates'])))
    for item in manifest['converted']:
        extra = ('  [' + '; '.join(item['warnings']) + ']') if item['warnings'] else ''
        print('  OK   {}  <- {} ({}, {} bit, p>q={}){}'.format(
            item['file'], item['source'], item['format'], item['bits'], item['p_gt_q'], extra))
    for item in manifest['skipped']:
        print('  SKIP {}: {}'.format(item['source'], item['reason']))
    for item in manifest['duplicates']:
        print('  DUP  {} (same modulus as {})'.format(item['source'], item['same_as']))
    print('manifest: {}'.format(os.path.join(args.out, 'manifest.json')))


if __name__ == '__main__':
    main()
