#!/usr/bin/env bash
#
# For licensing see accompanying LICENSE_MODEL file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
# Interactive FastVLM model downloader
# Allows selection of which models to download

set -e  # Exit on error

BASE_URL="https://ml-site.cdn-apple.com/datasets/fastvlm"
CHECKPOINTS_DIR="checkpoints"

# Available models
declare -A MODELS
MODELS["0.5b_stage2"]="llava-fastvithd_0.5b_stage2"
MODELS["0.5b_stage3"]="llava-fastvithd_0.5b_stage3"
MODELS["1.5b_stage2"]="llava-fastvithd_1.5b_stage2"
MODELS["1.5b_stage3"]="llava-fastvithd_1.5b_stage3"
MODELS["7b_stage2"]="llava-fastvithd_7b_stage2"
MODELS["7b_stage3"]="llava-fastvithd_7b_stage3"

# Helper functions
cleanup_on_error() {
    local zip_file="$1"
    if [ -f "$zip_file" ]; then
        echo "Cleaning up failed download: $zip_file"
        rm -f "$zip_file"
    fi
}

check_model_exists() {
    local model_name="$1"
    local model_dir="${CHECKPOINTS_DIR}/${model_name}"
    if [ -d "$model_dir" ] && [ "$(ls -A "$model_dir" 2>/dev/null)" ]; then
        return 0  # Model exists and is not empty
    fi
    return 1  # Model doesn't exist or is empty
}

download_and_extract() {
    local model_key="$1"
    local model_name="${MODELS[$model_key]}"
    local zip_file="${CHECKPOINTS_DIR}/${model_name}.zip"
    local model_dir="${CHECKPOINTS_DIR}/${model_name}"
    
    # Check if model already exists
    if check_model_exists "$model_name"; then
        echo "✓ Model $model_name already exists, skipping..."
        return 0
    fi
    
    # Download
    echo ""
    echo "Downloading $model_name..."
    if ! wget --progress=bar:force:noscroll --show-progress \
         -O "$zip_file" "${BASE_URL}/${model_name}.zip" 2>&1; then
        echo "❌ Failed to download $model_name"
        cleanup_on_error "$zip_file"
        return 1
    fi
    
    # Verify download
    if [ ! -f "$zip_file" ] || [ ! -s "$zip_file" ]; then
        echo "❌ Downloaded file is missing or empty: $zip_file"
        cleanup_on_error "$zip_file"
        return 1
    fi
    
    # Extract
    echo "Extracting $model_name..."
    cd "$CHECKPOINTS_DIR"
    
    # Use unzip with better error handling
    if ! unzip -q -o "$(basename "$zip_file")" 2>&1; then
        echo "❌ Failed to extract $model_name"
        cd - > /dev/null
        cleanup_on_error "$zip_file"
        return 1
    fi
    
    cd - > /dev/null
    
    # Verify extraction
    if ! check_model_exists "$model_name"; then
        echo "❌ Extraction failed: $model_name directory is missing or empty"
        cleanup_on_error "$zip_file"
        return 1
    fi
    
    # Clean up zip file only after successful extraction
    echo "Cleaning up zip file..."
    rm -f "$zip_file"
    
    echo "✓ Successfully downloaded and extracted $model_name"
    return 0
}

# Main script
main() {
    echo "FastVLM Model Downloader"
    echo "========================"
    echo ""
    echo "Available models:"
    echo "  1) 0.5B Stage 2 (intermediate checkpoint)"
    echo "  2) 0.5B Stage 3 (recommended for inference)"
    echo "  3) 1.5B Stage 2 (intermediate checkpoint)"
    echo "  4) 1.5B Stage 3 (recommended for inference)"
    echo "  5) 7B Stage 2 (intermediate checkpoint)"
    echo "  6) 7B Stage 3 (recommended for inference)"
    echo "  7) All Stage 3 models (recommended)"
    echo "  8) All models"
    echo ""
    
    read -p "Select models to download (comma-separated, e.g., 2,4,6 or 7): " selection
    
    # Create checkpoints directory
    mkdir -p "$CHECKPOINTS_DIR"
    
    # Parse selection
    IFS=',' read -ra SELECTIONS <<< "$selection"
    declare -a models_to_download=()
    
    for sel in "${SELECTIONS[@]}"; do
        sel=$(echo "$sel" | xargs)  # trim whitespace
        case "$sel" in
            1) models_to_download+=("0.5b_stage2") ;;
            2) models_to_download+=("0.5b_stage3") ;;
            3) models_to_download+=("1.5b_stage2") ;;
            4) models_to_download+=("1.5b_stage3") ;;
            5) models_to_download+=("7b_stage2") ;;
            6) models_to_download+=("7b_stage3") ;;
            7) models_to_download+=("0.5b_stage3" "1.5b_stage3" "7b_stage3") ;;
            8) models_to_download+=("0.5b_stage2" "0.5b_stage3" "1.5b_stage2" "1.5b_stage3" "7b_stage2" "7b_stage3") ;;
            *)
                echo "❌ Invalid selection: $sel"
                exit 1
                ;;
        esac
    done
    
    if [ ${#models_to_download[@]} -eq 0 ]; then
        echo "❌ No models selected"
        exit 1
    fi
    
    # Remove duplicates
    readarray -t unique_models < <(printf '%s\n' "${models_to_download[@]}" | sort -u)
    
    echo ""
    echo "Selected models:"
    for model in "${unique_models[@]}"; do
        echo "  - ${MODELS[$model]}"
    done
    echo ""
    
    read -p "Continue with download? [y/N]: " confirm
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        exit 0
    fi
    
    # Download and extract each model
    failed_models=()
    success_count=0
    
    for model_key in "${unique_models[@]}"; do
        if download_and_extract "$model_key"; then
            ((success_count++))
        else
            failed_models+=("${MODELS[$model_key]}")
        fi
    done
    
    # Summary
    echo ""
    echo "========================"
    echo "Download Summary"
    echo "========================"
    echo "Successfully downloaded: $success_count/${#unique_models[@]}"
    
    if [ ${#failed_models[@]} -gt 0 ]; then
        echo ""
        echo "Failed models:"
        for model in "${failed_models[@]}"; do
            echo "  - $model"
        done
        echo ""
        echo "You can re-run this script to retry failed downloads."
        exit 1
    else
        echo ""
        echo "✓ All models downloaded successfully!"
        echo ""
        echo "Downloaded models are in: $CHECKPOINTS_DIR/"
    fi
}

# Run main function
main "$@"
