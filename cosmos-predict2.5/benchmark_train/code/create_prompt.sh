#!/bin/bash

# Create metas folder if it doesn't exist
output_dir="../metas"
mkdir -p "$output_dir"

# Generate text files from 101.txt to 204.txt
for i in $(seq 101 204); do
    echo "Robot's left-hand fingers hold the bottle securely and move it into the box" > "$output_dir/${i}.txt"
done

echo "✅ Created text files 101.txt to 204.txt in $output_dir"
