#!/bin/bash

# Script to regenerate API documentation for pyhanami
# Requires: sphinx

set -e

# Configuration
SRC_DIR="../src"
DOCS_SOURCE_DIR="./source"
API_REFS_FILE="$DOCS_SOURCE_DIR/API-Reference.rst"
TEMP_DIR="/tmp/pyhanami_docs"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1"
}

# Check dependencies
check_dependencies() {
    local missing_deps=()
    
    if ! command -v sphinx-apidoc &> /dev/null; then
        missing_deps+=("sphinx")
    fi
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        error "Missing dependencies: ${missing_deps[*]}"
        echo "Install with: sudo apt-get install python3-sphinx"
        exit 1
    fi
}

# Generate API documentation
generate_docs() {
    log "Generating API documentation..."
    
    # Create temporary directory
    mkdir -p "$TEMP_DIR"
    
    # Generate RST files using sphinx-apidoc
    sphinx-apidoc -f -o "$TEMP_DIR" "$SRC_DIR" --separate
    
    # Create the consolidated API Reference RST file
    cat > "$API_REFS_FILE" << EOF
API Reference
=============

This document contains the complete API reference for pyhanami.

*Last update*: $(date)

EOF
    
    # Process each RST file and consolidate into single file
    find "$TEMP_DIR" -name "*.rst" -not -name "modules.rst" | sort | while read -r rst_file; do
        module_name=$(basename "$rst_file" .rst)
        
        echo "" >> "$API_REFS_FILE"
        echo "$(printf '=%.0s' {1..80})" >> "$API_REFS_FILE"
        echo "" >> "$API_REFS_FILE"
        
        # Append the content of each RST file
        cat "$rst_file" >> "$API_REFS_FILE"
        echo "" >> "$API_REFS_FILE"
    done
    
    # Cleanup
    rm -rf "$TEMP_DIR"
    
    log "API documentation generated successfully: $API_REFS_FILE"
    log "RST file is ready for Read the Docs integration"
}

# Main function
main() {
    # Change to script directory
    cd "$(dirname "$0")"
    
    # Validate directories
    if [ ! -d "$SRC_DIR" ]; then
        error "Source directory not found: $SRC_DIR"
        exit 1
    fi
    
    if [ ! -d "$DOCS_SOURCE_DIR" ]; then
        log "Creating docs source directory: $DOCS_SOURCE_DIR"
        mkdir -p "$DOCS_SOURCE_DIR"
    fi
    
    # Check dependencies
    check_dependencies
    
    case "${1:-}" in
        "--help"|"-h")
            echo "Usage: $0 [--help]"
            echo "  --help     Show this help message"
            echo ""
            echo "Generates consolidated RST API documentation from $SRC_DIR to $API_REFS_FILE"
            echo "File is ready for Read the Docs integration"
            ;;
        "")
            log "Generating API documentation..."
            generate_docs
            ;;
        *)
            error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
}

# Handle interrupts gracefully
trap 'log "Stopping..."; exit 0' INT TERM

# Run main function
main "$@"
