#!/bin/bash
# Modified from https://stackoverflow.com/a/75715924/3254683
# Export all drawio diagrams into PDF files. PDF files will have "sanitized" diagram names

set -e

if [ "$#" -eq 0 ]; then
    echo 1>&2 "Usage: $(basename "$0") DIAGRAM.drawio [OUTPUT_DIR]"
    exit 1
fi
if [[ ! "$1" == *.drawio ]]; then
    echo 2>&1 "Input file is not a .drawio file"
    exit 1
fi
drawio="${DRAWIO:-draw.io}"

diagram_file_name="$1"
# Allow optional target path as second argument; default remains src/figures
pdf_target_path="${2:-src/figures}"

# Ensure target directory exists
mkdir -p "$pdf_target_path"

while read page name; do
  $drawio --export --crop -t --format pdf --output "$pdf_target_path/$name.pdf" --page-index "$(expr $page - 1)" "$diagram_file_name"
done < <($drawio --export --format xml --output /dev/stdout --page-index 0 "$diagram_file_name" \
    | grep -Eo "name=\"[^\"]*" \
    | cut -c7- \
    | sed -e 's/[^a-zA-Z0-9_]/-/g' \
    | tr '[:upper:]' '[:lower:]' \
    | awk '{print NR,$1}')
