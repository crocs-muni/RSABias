"""
Packs PEM private keys into an rsabias dataset, so they can be used with dist, group, build and evaluate: one JSON
line per key (hexadecimal values) and a meta.json describing the source. Keys that cannot be read, or whose length
differs from the first key, are skipped with a warning.

Usage: python tools/pem2dataset.py OUT_DIR NAME VERSION KEY.pem... [--category Library] [--group 24]

--group is the group the keys are assumed to come from; evaluate needs it, the other tasks ignore it.
"""
import argparse
import json
import os
import sys

from Crypto.PublicKey import RSA


def main():
    parser = argparse.ArgumentParser(description='Pack PEM private keys into an rsabias dataset')
    parser.add_argument('out', help='output directory of the dataset')
    parser.add_argument('name', help='source name, e.g. OpenSSL')
    parser.add_argument('version', help='source version, e.g. 3.5.3')
    parser.add_argument('keys', nargs='+', help='PEM files with unencrypted RSA private keys')
    parser.add_argument('--category', default='Library', help='source type, e.g. Library, Card, HSM')
    parser.add_argument('--group', default=None, help='assumed group of the keys, needed by evaluate')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    bitlen = None
    records = 0
    with open(os.path.join(args.out, 'keys.json'), 'w') as f:
        for path in args.keys:
            try:
                with open(path) as key_file:
                    key = RSA.import_key(key_file.read())
            except (ValueError, IndexError, TypeError) as e:
                print('Skipping {}: {}'.format(path, e), file=sys.stderr)
                continue
            if not key.has_private():
                print('Skipping {}: not a private key'.format(path), file=sys.stderr)
                continue
            bitlen = bitlen or key.n.bit_length()
            if key.n.bit_length() != bitlen:
                print('Skipping {}: {} bits, the dataset has {}'.format(path, key.n.bit_length(), bitlen),
                      file=sys.stderr)
                continue
            values = {x: format(getattr(key, x), 'x') for x in 'nepqd'}
            f.write(json.dumps({'id': records, **values}) + '\n')
            records += 1
    if not records:
        sys.exit('No keys written')

    # No sha256 for the file: the reader only checks digests that are present
    meta = {'type': 'reference',
            'files': [{'name': 'keys.json', 'records': records}],
            'details': {'base_dict': {'id': 10, 'n': 16, 'e': 16, 'p': 16, 'q': 16, 'd': 16},
                        'bitlen': bitlen, 'category': args.category, 'compressed': False, 'fips_mode': False,
                        'format': 'json', 'name': args.name, 'version': args.version, 'group': args.group}}
    with open(os.path.join(args.out, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=4)
    print('Wrote {} keys of {} bits to {}'.format(records, bitlen, args.out))


if __name__ == '__main__':
    main()
