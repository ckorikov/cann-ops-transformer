#!/bin/bash
# This script builds the operator and installs a python torch extension package 'select_attn_decoding_ops'

# build operator as shared lib (.so file)
bash ./compile.sh

# build torch extension
rm -rf build select_attn_decoding_ops.egg-info
pip uninstall -y select_attn_decoding_ops
pip install . --no-build-isolation