Pre-trained ELMo Representations for Many Languages
===================================================

## Pre-requirements

* **must** python >= 3.6 (if you use python3.5, you will encounter this issue https://github.com/HIT-SCIR/ELMoForManyLangs/issues/8)
* pytorch 0.4
* other requirements from allennlp

### Install the package

~You need to install the package to use the embeddings with the following commends~
```
python setup.py install
```

### Use in command line

```
$ python -m elmo test \
    --input_format conll \
    --input /path/to/your/input \
    --model /path/to/your/model \
    --output_prefix /path/to/your/output \
    --output_format hdf5 \
    --output_layer -1
```

## Citation

If our ELMo gave you nice improvements, please cite us.

```
@InProceedings{che-EtAl:2018:K18-2,
  author    = {Che, Wanxiang  and  Liu, Yijia  and  Wang, Yuxuan  and  Zheng, Bo  and  Liu, Ting},
  title     = {Towards Better {UD} Parsing: Deep Contextualized Word Embeddings, Ensemble, and Treebank Concatenation},
  booktitle = {Proceedings of the {CoNLL} 2018 Shared Task: Multilingual Parsing from Raw Text to Universal Dependencies},
  month     = {October},
  year      = {2018},
  address   = {Brussels, Belgium},
  publisher = {Association for Computational Linguistics},
  pages     = {55--64},
  url       = {http://www.aclweb.org/anthology/K18-2005}
}
```

Please also cite the
[NLPL Vectors Repository](http://wiki.nlpl.eu/index.php/Vectors/home)
for hosting the models.
```
@InProceedings{fares-EtAl:2017:NoDaLiDa,
  author    = {Fares, Murhaf  and  Kutuzov, Andrey  and  Oepen, Stephan  and  Velldal, Erik},
  title     = {Word vectors, reuse, and replicability: Towards a community repository of large-text resources},
  booktitle = {Proceedings of the 21st Nordic Conference on Computational Linguistics},
  month     = {May},
  year      = {2017},
  address   = {Gothenburg, Sweden},
  publisher = {Association for Computational Linguistics},
  pages     = {271--276},
  url       = {http://www.aclweb.org/anthology/W17-0237}
}
```
