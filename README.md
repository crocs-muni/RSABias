# Biased RSA private keys: Origin attribution of GCD-factorable keys

Python tool for black-box analysis of RSA key generation in cryptographic libraries and for RSA key classification. This tool accompanies the paper [Biased RSA private keys: Origin attribution of GCD-factorable keys](TBA) presented at [ESORICS 2020](https://www.surrey.ac.uk/esorics-2020) conference. 

Using this project, you can audit the origin of your RSA keypair. Also, it is possible to replicate our work and analyze large datasets of keys. Beware that when classifying one key, the process is not bulletproof, as our model achieves 47.7% accuracy (on 26 classes). 

## Install

The code is tested on Python 3.6. It should suffice to clone and install using the setup script. Note that since the stable part of the code is tied to the experimental part, this package is not distributed as a pip package. Note that the required gmpy2 package has some [install requirements](https://gmpy2.readthedocs.io/en/latest/intro.html#installation). The full chain of commands is:

```
git clone git@github.com:matusn/RSABias.git &&
cd RSABias &&
python3 -m venv venv &&
source venv/bin/activate &&
python3 ./setup.py install
```

## Usage

### Classification of a single key

From the repository root, you can try to classify the sample key as:

```bash
rsabias -a classify -i rsabias/model/sample_private_key.pem -o .
```

where `-i` parameter denotes a path to the RSA private key in DER or PEM format, and `-o` parameter denotes a path to the folder where the classification report will be stored. Consequently, running the command above should print basic information to the standard output:

```bash
Your key with sha-256 digest: e10d8b68b1e2b595f69813a99524d9dce0c0f80b092f9466d80b587f587262ba gets classified as:
	- Classification group: 24
	- Score: 0.023450350877192983
	- If a key is marked as coming from Group 24, the model is in 90.21% cases right.
	- The model correctly classifies 96.83% keys that are actually from Group 24.
	- This group contains the following sources:
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

### Advanced usage

A bunch of distinct tasks are available for full analysis of the RSA keys. This section of the readme is not meant to be self-contained, but rather provide a guidance. The user will most probably need to dive into the code when aiming for replication of the experiment. A list of available tasks with short description follows (with the assumption that the user is aknowledged with out paper). The exact list of expected parameters for each of the tasks can be simply read out from the [__main__.py](TBA). The tasks are presented in the order that leads to experiment replication.

- `convert` -  Since we internally support multiple formats of the datasets, this task is capable of converting between the formats (csv -> json, enabling compression, etc.)
- `dist+plot` - Computes the exact distribution of the selected features on a whole dataset. Out of these features, labels for the dataset are constructed. This task also plots a dendrogram of the classes, showing how similar/distant the keys from the respective sources are. 
- `group` - Performs the clustering task, using the distributions of the features as an input. 
- `split` - Splits the dataset into training and test part. Out of each source, at least 10 000 keys are taken into the test set as default.  
- `filter` - Prepares the test dataset. This task shuffles and merges the keys into a single artifically created dataset, that can be fed into tasks below.
- `build` - Based on a list of transformations, groups obtained from clustering, and a dataset, this task constructs the classification tables. Three Bayes classifiers can be selected: naive, complex, and cross-feature classifier. See Section 3 of paper for more information. 
- `evaluate` - Evaluates the classifier on a dataset of keys. Produces several json files with the model perormance report (if the labels are available). 
- `batch_gcd` - Builds specific classification tables for the GCD-factorable dataset (that only use single prime) and directly uses them to classify the whole dataset. 
- `visualize` - To-be-called after the `evaluate` task. Takes the model performance results and prints them into a table. More info at [classification_table_template](https://github.com/matusn/RSABias/tree/master/classification_table_template).
- `classify` - Used to classify a single key. This is a main task that was already described above. 

Each of the task can be invoked by calling

```bash
rsabias -a task_name task_arguments
```

The parameters of the tasks are:

- `-a` or `--action`: which action to perform, see above.
- `-t` or `--trans`: path to the file with transformations, see [model](TBA#transformations) for more information.
- `-i` or `--inp`: path to the input, whatever the task needs.
- `-o` or `--out`: path to the output, whatever the task produces.
- `-f` or `--format`: format of the dataset (csv, json). 
- `-g` or `--groups`: path to the file with groups, see [model](TBA#groups) for more information.
- `-d` or `--decompress`: whether to compress the dataset or not.
- `-s` or `--subspaces`: Subspaces for the clustering task.
- `-c` or `--classtable`: path to the classification table.
- `-m` or `--method`: What model to use. naive bayess or complex Bayess.
- `-l` or `--labels`: Whether the labels for the dataset are at hand, used for evaluation of the model performance.
- `-p` or `--prime_wise`: Whether single-prime model should be used, for gcd-factorized key classification. 
- `-r` or `--remove_duplicities`: Whether duplicate keys should be deleted from a dataset.

To summarize, in order to fully replicate our experiments, one must do:

1. Download our datasets (see below).
2. Use our [feature vector](TBA) to count distribution of the features on the whole dataset. (`dist+plot` task). The output of this task can be found at [model/distributions](TBA) folder.
3. Split the key sources into multiple groups with the help of automated clustering. (`group` task). The output of this task can be found at [model/groups](TBA) folder.
4. Split and filter the dataset into training and test part (`split` and `filter` tasks)
5. Build the model. (`build` task)
6. Evaluate the performance of the model. (`evaluate` task)
7. Possibly classify the [batch-gcd dataset](TBA). (`batch-gcd` task)
8. Possibly visualize the model performance. (`visualize` task)

We present further description of the utilized data structures (groups, features, ...) in the [model folder](https://github.com/matusn/RSABias/tree/master/model).

## Dataset

### Train & Test

We collected, analyzed, and published the largest dataset of RSA keys with a known origin from 70 libraries (43 open-source libraries, 5 black-box libraries, 3 HSMs, 19 smartcards). We both expanded the datasets from previous work and generated new keys from additional libraries for the sake of this study.

We are primarily interested in 2048-bit keys, what is the most commonly used key length for RSA. As in previous studies, we also generate shorter keys (512 and 1024 bits) to speed up the process, while verifying that the chosen biased features are not influenced by the key size. This makes the keys of different sizes interchangeable for the sake of our study. We assume that repeatedly running the key generation locally approximates the distributed behaviour of many instances of the same library. 

The dataset of more than 160 million RSA keys can be accessed from [Google Drive](https://drive.google.com/drive/folders/0B0PpUrsKytcyMllkUHJ0RkZkdzA?usp=sharing).

### GCD-factorable keys

We utilized the training data to classify a [Rapid-7](https://opendata.rapid7.com/sonar.ssl/) dataset of RSA keys factorable by batch-GCD method. More on this dataset in [BatchGCD-primes-only](https://github.com/matusn/RSABias/tree/master/BatchGCD-primes-only). Using this dataset, we classified more than 80 thousand of weak RSA keys. 

## Paper abstract

The paper can be found at: [arxiv.com/linkToPaper](TBA). 

In 2016, Švenda et al. (USENIX 2016, The Million-key Question) reported that the implementation choices in cryptographic libraries allow for qualified guessing about the origin of public RSA keys.
We extend the technique to two new scenarios when not only public but also private keys are available for the origin attribution -- analysis of a source of GCD-factorable keys in IPv4-wide TLS scans and forensic investigation of an unknown source. We learn several representatives of the bias from the private keys to train a model on more than 150 million keys collected from 70 cryptographic libraries, hardware security modules and cryptographic smartcards. Our model not only doubles the number of distinguishable groups of libraries (compared to public keys from Švenda et al.) but also improves more than twice in accuracy w.r.t. random guessing when a single key is classified. For a forensic scenario where at least 10 keys from the same source are available, the correct origin library is correctly identified with average accuracy of 89\% compared to 4\% accuracy of a random guess. The technique was also used to identify libraries producing GCD-factorable TLS keys, showing that only three groups are the probable suspects.

## Project status & Contributing

This project is a result of a contiuous efforts of the [CRoCS](crocs.fi.muni.cz) laboratory that resulted into following publications:

- [The Million-Key Question – Investigating the Origins of RSA Public Keys](https://crocs.fi.muni.cz/public/papers/usenix2016) - USENIX 2016, Best Paper Award,
- [The Return of Coppersmith's Attack: Practical Factorization of Widely Used RSA Moduli](https://crocs.fi.muni.cz/public/papers/rsa_ccs17) - ACM CCS 2017,
- [Measuring Popularity of Cryptographic Libraries in Internet-Wide Scans](https://crocs.fi.muni.cz/public/papers/acsac2017) - ACSAC 2017,
- [Biased RSA private keys: Origin attribution of GCD-factorable keys](https://crocs.fi.muni.cz/public/papers/privrsa_esorics20) - ESORICS 2020.

If you would like to contribute, feel free to [Adam Janovsky](https://github.com/adamjanovsky) or open an issue.

## Authors

[Adam Janovsky](https://github.com/adamjanovsky), [Matus Nemec](https://github.com/matusn), [Petr Svenda](https://github.com/petrs), [Peter Sekan](https://github.com/psekan) and [Vashek Matyas](matyas.cz); all from [Masaryk University](muni.cz), [Faculty of Informatics](fi.muni.cz), [Center for Research on Cryptography and Security](crocs.fi.muni.cz) in Czech Republic.

## License

This tool is to be available under [MIT License](https://github.com/matusn/RSABias/blob/master/LICENSE).
