#!/bin/bash

# This script orchestrates the entire data preparation process by calling the
# `prepare_data_from_source.py` script for each of the original Kaggle datasets.
# It builds a single, consolidated CSV file ready for model training.

# --- Configuration ---
# The final output file that will contain the consolidated data.
OUTPUT_CSV="data/processed/final_dataset.csv"

# The directory where the original Kaggle CSVs are stored.
SOURCE_DIR="data/original_kaggle_datasets"

# --- Start of Script ---
echo "Starting data preparation process..."

# 1. Clean up any previous runs to ensure a fresh start.
echo "Removing old output file: $OUTPUT_CSV"
rm -f $OUTPUT_CSV

# 2. Process Kaggle Dataset 1: Depression vs. Neutral
# Extracts posts where 'is_depression' is 1 and labels them as 'Depression'.
echo -e "\n--- Processing Kaggle Dataset 1: Depression ---"
python prepare_data_from_source.py \
    --input-csv "$SOURCE_DIR/kaggle_1.csv" \
    --text-columns "clean_text" \
    --filter-column "is_depression" \
    --filter-value "1" \
    --output-csv "$OUTPUT_CSV" \
    --output-label "Depression"

# Extracts posts where 'is_depression' is 0 and labels them as 'Neutral'.
echo -e "\n--- Processing Kaggle Dataset 1: Neutral ---"
python prepare_data_from_source.py \
    --input-csv "$SOURCE_DIR/kaggle_1.csv" \
    --text-columns "clean_text" \
    --filter-column "is_depression" \
    --filter-value "0" \
    --output-csv "$OUTPUT_CSV" \
    --output-label "Neutral"

# 3. Process Kaggle Dataset 2: Depression Only
# Extracts all posts and labels them as 'Depression'. No filtering needed.
echo -e "\n--- Processing Kaggle Dataset 2: Depression ---"
python prepare_data_from_source.py \
    --input-csv "$SOURCE_DIR/kaggle_2.csv" \
    --text-columns "text" \
    --output-csv "$OUTPUT_CSV" \
    --output-label "Depression"

# 4. Process Kaggle Dataset 3: Suicide vs. Neutral
# Extracts posts where 'class' is 'suicide' and labels them as 'Suicide'.
echo -e "\n--- Processing Kaggle Dataset 3: Suicide ---"
python prepare_data_from_source.py \
    --input-csv "$SOURCE_DIR/kaggle_3.csv" \
    --text-columns "text" \
    --filter-column "class" \
    --filter-value "suicide" \
    --output-csv "$OUTPUT_CSV" \
    --output-label "Suicide"

# Extracts posts where 'class' is 'non-suicide' and labels them as 'Neutral'.
echo -e "\n--- Processing Kaggle Dataset 3: Neutral ---"
python prepare_data_from_source.py \
    --input-csv "$SOURCE_DIR/kaggle_3.csv" \
    --text-columns "text" \
    --filter-column "class" \
    --filter-value "non-suicide" \
    --output-csv "$OUTPUT_CSV" \
    --output-label "Neutral"

echo -e "\nData preparation complete. The consolidated dataset is available at: $OUTPUT_CSV"