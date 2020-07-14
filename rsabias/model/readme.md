# Key classification model

This folder contains a single classification model used in our study. The model is a complex Bayes classifier built over the [5p_5q_blum_mod_roca.json](https://github.com/crocs-muni/RSABias/blob/master/rsabias/model/transformations/5p_5q_blum_mod_roca.json) feature vector. This document also provides a consise explanation of how we define the groups of the RSA keys (labels or classes in ML terminology) and how we represent our transformations (features in ML terminology).

## Bayes classifier

The model is provided in a form of a pickle object -- [classification_table_complex.pkl](https://github.com/crocs-muni/RSABias/blob/master/rsabias/model/classification_table_complex.pkl) -- that represents  an instance of `ProbaTable` class in the [classifier](https://github.com/crocs-muni/RSABias/blob/master/rsabias/core/classifier.py) module. The model can be directly used with our tool as a command-line argument (see readme in the root of the repository).

If the developer aims to reconstruct the model by himself, it can be done by calling

```python
table = classifier.ProbaTable(trans_path, groups_json_path, method)
table.init_common_structures()
table.load_classification_tables(table_path, method)
```

where

- `trans_path` is a path to [5p_5q_blum_mod_roca.json](https://github.com/crocs-muni/RSABias/blob/master/rsabias/model/transformations/5p_5q_blum_mod_roca.json).
- `groups_json_path` is a path to [groups.json](https://github.com/crocs-muni/RSABias/blob/master/rsabias/model/groups/groups.json).
- `method` use a string `complex`.

Then, the model should reside in the `table` variable. Further, it is possible to build the [evaluator](https://github.com/crocs-muni/RSABias/blob/master/rsabias/core/classifier.py) (that runs over a dataset) object with the `table` variable. Using the `Evaluator` object, many interesting tasks can be called that can result the whole dataset.

## Groups description

The groups, demonstrated in [groups.json](https://github.com/crocs-muni/RSABias/blob/master/rsabias/model/groups/groups.json), illustrate the clustering task (discussed in [the paper](TBA), section 2.3). Put simply, `groups.json` are a dictionary of lists. The key of each dictionary is a name of the group. Its values (a list) form the sources that produce very similar RSA keys. For instance, see the example below:

```json
{
"16": [
        "HSM Utimaco Security Server Se50 LAN Appliance 1024",
        "HSM Utimaco Security Server Se50 LAN Appliance 512"
    ],
"11": [
    "HSM SafeNet Luna SA-1700 LAN 2048",
    "Card Feitian JavaCOS A22 1024",
    "Card Feitian JavaCOS A22 512",
    "Card Feitian JavaCOS A40 1024",
    "Card Feitian JavaCOS A40 512",
    "Card Oberthur Cosmo 64 1024",
    "Library cryptlib 3.4.3.1 1024",
    "Library cryptlib 3.4.3.1 2048",
    "Library cryptlib 3.4.3.1 512",
    "Library cryptlib 3.4.3 1024",
    "Library cryptlib 3.4.3 2048",
    "Library cryptlib 3.4.3 512"
    ]
}
```

This segment of `groups.json` illustrates that both `Oberthur Cosmo 64` and `Feitian JavaCOS A40` produce similarly distributed RSA keys, whereas the `Ultimatico Security Server` produces different RSA keys. 

Also, the `groups` folder contains a dendrogram of our dataset, displaying all 26 distinct groups of keys that we are able to tell apart (they represent classes of our classifier). 


## Transformations description

This directory contains description of transformations (or features) that were applied in our study. More specifically, three feature vectors are given as examples. Those are:

- `5p_5q_blum_mod_roca.json` - the most reliable feature vector, extracting 5 features from both primes`p`, and `q`
- `prime_wise.json` - 5 features extracted only from a single prime. Used to classify batches of GCD-factorized keys.
- `private_simple.json` - an example of feature vector, not really used in our models.

The source file [features.py](https://github.com/crocs-muni/RSABias/blob/master/rsabias/core/features.py) lists all viable options for the features. Yet, a developer is welcomed to write his own features and introduce them into the codebase. The individual features are then combined into a *feature vector*. The *feature vector* is represented as a `.json` file and applied to every key in the dataset to construct a model. A feature vector is an inherent part of the model and must accompany most of the task specifications. 

The distributions of the feature vector on the whole dataset are pre-computed in the [model/distribution](https://github.com/crocs-muni/RSABias/tree/master/rsabias/model/distributions) folder.

The rest of this document illustrates how to read the json files.

### 5p_5q_blum_mod_roca.json

The individual features are chained into a dictionary that then forms the json file. Example of such feature may be *"extract 5 most significant bits of prime p from the key"*. Such feature is depicted as:

```json
"transformation": "MostSignificantBits",
    "options": {
  	"input": "{p}",
        "skip": 0,
        "count": 5,
        "byte_aligned": true
  }
```

The exact feature vector captured by `5p_5q_blum_mod_roca.json` is described in the [paper](TBA), Section 2.2.

## Distribution description

The folder distribution list the complete probability distribution on the feature vector `5p_5q_blum_mod_roca.json` of our dataset. For each of the sources and for every encountered feature vector instance, a number of keys is displayed in the respective json file. For instance, 512b keys of Athena IDProtect card exhibit 499 keys with the feature vector `(24, 27, False, 1, False)`, as depicted by [dist.json](https://github.com/crocs-muni/RSABias/blob/master/rsabias/model/distributions/Card/Athena/IDProtect/512/dist.json) on line 5:

```json
"(24, 27, False, 1, False)": 499,
```

Out of these distributions, our Bayes classifier is built. The user that aims to replicate our experiment should thus obtain the very same distribution on our dataset which can serve as a sanity check. 
