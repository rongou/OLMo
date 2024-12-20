#!/bin/bash

# Check if correct number of arguments provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 config.yaml data_root"
    exit 1
fi

config_file="$1"
data_root="$2"

# Check if config file exists
if [ ! -f "$config_file" ]; then
    echo "Error: Config file $config_file does not exist"
    exit 1
fi

# Create data root directory if it doesn't exist
mkdir -p "$data_root"

# Function to download a single URL
download_url() {
    local url="$1"
    local data_root="$2"

    # Extract the path after olmo-data.org
    local relative_path="${url#https://olmo-data.org/}"
    local target_dir="$data_root/$(dirname "$relative_path")"
    local target_file="$data_root/$relative_path"

    echo "Downloading ${url}"

    # Create target directory
    mkdir -p "$target_dir"

    # Download file with wget
    wget -c -q "$url" -O "$target_file"

    echo "Done downloading ${url}"
}

export -f download_url

# Extract all URLs and pipe them to parallel
grep -o 'https://olmo-data\.org/[^"]*' "$config_file" | \
    parallel -j 8 download_url {} "$data_root"

echo "All downloads complete!"