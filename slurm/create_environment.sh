#!/bin/bash

micromamba create --copy -n olmo\
 python=3.12\
 pytorch[cuda]\
 triton\
 flash-attn\
 "numpy<2.0.0"\
 fsspec=2024.9.0

micromamba activate olmo
pip install -e .[all]
