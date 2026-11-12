"""
rsa_fingerprint.py re-implements the 5p_5q_blum_mod_roca features without the package; both must agree. The
xfail tests document known bugs of rsa_fingerprint.py and start failing once they are fixed.
"""
import pytest
from Crypto.PublicKey import RSA

import pipeline
from rsabias.core import classifier, key

# rsa_fingerprint reports the smallest bound below which p-1 has a divisor (-1 for none), the package the largest
# bound below which it has none (1 for none).
TO_PACKAGE_MODULAR = {5: 1, 251: 5, 17863: 251, -1: 17863}


def fingerprints():
    import rsa_fingerprint
    trans = classifier.trans_from_path(pipeline.TRANS)
    result = []
    for path in pipeline.key_files():
        with open(path) as f:
            pem = f.read()
        k = RSA.import_key(pem)
        fingerprint = rsa_fingerprint.RSAFingerprint('%x' % k.p, '%x' % k.q, '%x' % k.n)
        fingerprint.compute_fingerprint()
        result.append((pipeline.key_name(path), fingerprint, trans.apply(key.Key.import_standard_format(pem))))
    return result


def test_msb5():
    for name, fp, (msb5p, msb5q, _, _, _) in fingerprints():
        assert (fp.msb5_p, fp.msb5_q) == (msb5p, msb5q), name


def test_roca():
    for name, fp, (_, _, _, _, roca) in fingerprints():
        assert fp.is_roca == roca, name


@pytest.mark.xfail(strict=True, raises=AssertionError, reason='is_blum is never set to True')
def test_blum():
    for name, fp, (_, _, blum, _, _) in fingerprints():
        assert fp.is_blum == blum, name


@pytest.mark.xfail(strict=True, raises=AssertionError, reason='small divisor search includes even numbers, 4 divides p-1 when p = 1 mod 4')
def test_small_divisors():
    for name, fp, (_, _, _, modular, _) in fingerprints():
        assert min(TO_PACKAGE_MODULAR[fp.small_divisors_p], TO_PACKAGE_MODULAR[fp.small_divisors_q]) == modular, name


@pytest.mark.xfail(strict=True, raises=AssertionError, reason='RSAFingerprintSet keeps its lists as class attributes')
def test_fingerprint_sets_independent():
    import rsa_fingerprint
    first, second = rsa_fingerprint.RSAFingerprintSet([]), rsa_fingerprint.RSAFingerprintSet([])
    assert first.pqn_fingerprints is not second.pqn_fingerprints
