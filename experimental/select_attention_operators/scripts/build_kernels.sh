#!/bin/bash

# Iterate over all subdirectories in kernels/ and build each
for dir in kernels/*/; do
    echo "Building kernel in: $dir"
    pushd "$dir"
    # Check if build.sh exists and is executable, then run it
    if [ -f "./build.sh" ]; then
        chmod +x ./build.sh  # Ensure it's executable
        ./build.sh
    else
        echo "Warning: build.sh not found in $dir"
    fi
    popd
done