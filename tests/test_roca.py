"""
None of the real test keys is ROCA-vulnerable, so a broken ROCA check would go unnoticed. These keys are synthetic
512-bit keys with the structure of the vulnerable Infineon RSALib primes, p = k * M + (65537^a mod M), where M = 167#
is the primorial the detectors use for 512-bit moduli. Like the Infineon keys of the training data, the five most
significant bits of each prime are 24 or 25 (0b1100x).
"""
import math
import os
import random

import pytest
import sympy
from Crypto.PublicKey import RSA

import pipeline
from rsabias.core import classifier, key

PRIME_BITS = 256
M = sympy.primorial(167, nth=False)


def roca_prime(rng):
    low, high = 24 << (PRIME_BITS - 5), 26 << (PRIME_BITS - 5)
    while True:
        p = rng.randrange(low // M, high // M) * M + pow(65537, rng.randrange(1, M), M)
        if low <= p < high and sympy.isprime(p):
            return p


@pytest.fixture(scope='module')
def roca_keys(tmp_path_factory):
    """Paths of PEM files with synthetic ROCA keys; deterministic."""
    rng = random.Random(2017)
    out = tmp_path_factory.mktemp('roca_keys')
    paths = []
    for i in range(3):
        p, q = roca_prime(rng), roca_prime(rng)
        e = 65537
        pem = RSA.construct((p * q, e, pow(e, -1, math.lcm(p - 1, q - 1)), p, q)).export_key('PEM')
        path = out / 'roca_{}.pem'.format(i)
        path.write_bytes(pem)
        paths.append(str(path))
    return paths


def modulus(path):
    with open(path) as f:
        return RSA.import_key(f.read()).n


def package_roca_feature(path):
    trans = classifier.trans_from_path(pipeline.TRANS)
    with open(path) as f:
        return trans.apply(key.Key.import_standard_format(f.read()))[-1]


def test_package_feature(roca_keys):
    assert all(package_roca_feature(p) for p in roca_keys)
    assert not any(package_roca_feature(p) for p in pipeline.key_files())


def test_roca_py(roca_keys):
    import roca
    fingerprinter = roca.RocaFingerprinter()
    assert all(fingerprinter.has_fingerprint_moduli(modulus(p)) for p in roca_keys)
    assert not any(fingerprinter.has_fingerprint_moduli(modulus(p)) for p in pipeline.key_files())


def test_rsa_fingerprint(roca_keys):
    import rsa_fingerprint

    def is_roca(path):
        with open(path) as f:
            k = RSA.import_key(f.read())
        fingerprint = rsa_fingerprint.RSAFingerprint('%x' % k.p, '%x' % k.q, '%x' % k.n)
        fingerprint.compute_fingerprint()
        return fingerprint.is_roca

    assert all(is_roca(p) for p in roca_keys)
    assert not any(is_roca(p) for p in pipeline.key_files())


def test_attributed_to_infineon_only(roca_keys, tmp_path):
    """In the model, only the Infineon group has keys with the ROCA fingerprint."""
    for i, path in enumerate(roca_keys):
        out = tmp_path / str(i)
        out.mkdir()
        report = pipeline.load_json(pipeline.classify(path, str(out)))
        assert [g for g, r in report.items() if r['score'] > 0] == ['1']
        assert 'Card Infineon JTOP 80K 512' in report['1']['sources']
