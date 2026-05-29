# Data

The full geographic datasets are not committed to this repository. Place known and unseen graph folders under this directory before training.

Expected layout:

```text
data/region_v6/train.txt
data/region_v6/valid.txt
data/region_v6/test.txt
data/region_6_ind/train.txt
data/region_6_ind/valid.txt
data/region_6_ind/test.txt
```

Each file contains one triple per line, using `*` or `^` as the separator:

```text
<head>*<relation>*<tail>
```

`sample_region_v6` and `sample_region_6_ind` provide small examples for checking preprocessing and attribute extraction.