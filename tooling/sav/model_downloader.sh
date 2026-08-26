#!/bin/sh

# Downloads the models (in parts) from GitHub Releases and stitches them back together.
# Files were generated with these commands and then manually uploaded:
# `sha256sum <model>.pkl  > <model>.pkl.sha256
# `split -b 1900M -d <model>.pkl <model>.pkl.part`

set -eu

BASE=https://github.com/PEOPLE-ECCO/ecco-marine-solutions/releases/download/v1.0.0

fetch() {
  name=$1; parts=$2
  : > "$name"
  i=0
  while [ "$i" -lt "$parts" ]; do
    full="$BASE/$name.part$(printf '%02d' $i)"
    echo "Downloading $full"
    curl -fsSL --retry 5 --retry-all-errors $full >> "$name"
    i=$((i + 1))
  done
}

verify() {
  curl -fsSL "$BASE/$1.sha256" -o "$1.sha256"
  sha256sum -c "$1.sha256"
  rm -f "$1.sha256"
}

fetch rf_model_Malaysia.pkl 1
fetch rf_model_PEI.pkl 3

verify rf_model_Malaysia.pkl
verify rf_model_PEI.pkl
