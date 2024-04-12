#!/usr/bin/env bash

set -e

read -r -p "Rerun Opera + ablation evaluation? [y/N] " response
response=${response,,}
if [[ "$response" =~ ^(yes|y)$ ]]
then
    pushd opera
    bash ./run_opera.sh
    bash ./run_opera_nosymb.sh
    bash ./run_opera_nodecomp.sh
    popd
fi

read -r -p "Rerun CVC5 evaluation? [y/N] " response
response=${response,,}
if [[ "$response" =~ ^(yes|y)$ ]]
then
    pushd cvc5
    bash run.sh
    bash run_nexmark.sh
    popd
fi

read -r -p "Rerun Sketch evaluation? [y/N] " response
response=${response,,}
if [[ "$response" =~ ^(yes|y)$ ]]
then
    pushd sketch
    bash run.sh
    bash run_nexmark.sh
    popd
fi

read -r -p "Run evaluation script? [Y/n] " response
response=${response,,}
if [[ "$response" =~ ^(no|n)$ ]]
then
    exit 0
fi

python eval.py
