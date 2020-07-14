## Rapid-7 dataset

This folder contains our internal representation of a [Rapid7 dataset](https://opendata.rapid7.com/sonar.ssl/). This dataset contains already (GCD) factorized keys divided into two groups:

1. The first group captures the keys that exhibit an OpenSSL fingerprint -- such keys reliably originate from the OpenSSL library. A sanity check can be performed on those keys. In exact, our model should correctly classify them as OpenSSL keys.
2. The second group captures the keys that *do not* exhibit an OpenSSL fingerprint. Consequently, these keys do not originate from the OpenSSL library and are the main subject of our [study](TBA our paper). For the classification results, see Section 5 of [the paper](TBA).  
