# visrep

This repository an extension of [fairseq](https://github.com/pytorch/fairseq) to enable training with visual text representations. 

For more information, please see:
- [Salesky et al. (2021): Robust Open-Vocabulary Translation from Visual Text Representations.](https://arxiv.org/abs/2104.08211) In *Proceedings of EMNLP 2021*.

## Overview 



## Installation

The installation is the same as [fairseq](https://github.com/pytorch/fairseq), plus additional requirements specific to visual text.

**Requirements:**
* [PyTorch](http://pytorch.org/) version >= 1.5.0
* Python version >= 3.6
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)

**To install and develop locally:**
``` bash
git clone https://github.com/esalesky/visrep
cd visrep
pip install --editable ./
pip install -r examples/visual_text/requirements.txt
```

## Training 


## Inducing noise

We induced five types of noise, as below:
- **swap**: swaps two adjacent characters per token. applies to words of length >=2 *(Arabic, French, German, Korean, Russian)*
- **cmabrigde**: permutes word-internal characters with first and last character unchanged. applies to words of length >=4 *(Arabic, French, German, Korean, Russian)*
- **diacritization**: diacritization, applied via [camel-tools](https://github.com/CAMeL-Lab/camel_tools) *(Arabic)*
- **unicode**: substitutes visually similar Latin characters for Cyrillic characters *(Russian)*
- **l33tspeak**: substitutes numbers or other visually similar characters for Latin characters *(French, German)*

The scripts to induce noise are in [scripts/visual_text](https://github.com/esalesky/visrep/tree/main/scripts/visual_text), where -p is the probability of inducing noise per-token, and can be run as below.  
In our paper we use p from 0.1 to 1.0, in intervals of 0.1.

```
cat test.de-en.de | python3 scripts/visual_text/swap.py -p 0.1 > visual/test-sets/swap_10.de-en.de
cat test.ko-en.ko | python3 scripts/visual_text/cmabrigde.py -p 0.1 > visual/test-sets/cam_10.ko-en.ko
cat test.ar-en.ar | python3 scripts/visual_text/diacritization.py -p 0.1 > visual/test-sets/dia_10.ar-en.ar
cat test.ru-en.ru | python3 scripts/visual_text/cyrillic_noise.py -p 0.1 > visual/test-sets/cyr_10.ru-en.ru
cat test.fr-en.fr | python3 scripts/visual_text/l33t.py -p 0.1 > visual/test-sets/l33t_10.fr-en.fr
```

## License

fairseq(-py) is MIT-licensed.

## Citation

Please cite as:

``` bibtex
@inproceedings{salesky-etal-2021-robust,
    title = "Robust Open-Vocabulary Translation from Visual Text Representations",
    author = "Salesky, Elizabeth  and
      Etter, David  and
      Post, Matt",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/2104.08211",
}

@inproceedings{ott2019fairseq,
  title = {fairseq: A Fast, Extensible Toolkit for Sequence Modeling},
  author = {Myle Ott and Sergey Edunov and Alexei Baevski and Angela Fan and Sam Gross and Nathan Ng and David Grangier and Michael Auli},
  booktitle = {Proceedings of NAACL-HLT 2019: Demonstrations},
  year = {2019},
}
```
