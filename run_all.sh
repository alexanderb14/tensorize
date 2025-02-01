#!/bin/sh
set -x
set -e

# Build the images
docker build -t tensorize-artifact -f Dockerfile .

# Run the experiments
mkdir -p out
docker run -v $(pwd)/out:/root/out -it tensorize-artifact "python benchmark/run.py"

# Plot the results
docker run -v $(pwd)/out:/root/out -it tensorize-artifact "Rscript benchmark/plot.R"