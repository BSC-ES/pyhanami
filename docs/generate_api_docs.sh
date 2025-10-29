#!/bin/bash

# Script to regenerate API documentation for pyhanami
# Requires: sphinx

set -e

# Configuration
SRC_DIR="../src/pyhanami/diags"
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
    
    if ! command -v sphinx-build &> /dev/null; then
        missing_deps+=("sphinx")
    fi
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        error "Missing dependencies: sphinx"
        echo "Install with: sudo apt-get install python3-sphinx"
        exit 1
    fi
}

# Generate API documentation
generate_docs() {
    log "Generating API documentation..."
    
    # Create temporary directory
    mkdir -p "$TEMP_DIR"
    
    # Generate RST files using sphinx-apidoc with full content
    sphinx-apidoc -f -e -o "$TEMP_DIR" "$SRC_DIR" --separate
    
    # Create a temporary conf.py for sphinx-build
    cat > "$TEMP_DIR/conf.py" << EOF
import os
import sys
sys.path.insert(0, os.path.abspath('$(realpath "$SRC_DIR")'))

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.viewcode', 'sphinx.ext.napoleon']
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}
master_doc = 'index'
EOF
    
    # Use sphinx-build to generate the actual documentation
    if ! sphinx-build -b text "$TEMP_DIR" "$TEMP_DIR/output" 2>/dev/null; then
        warn "Sphinx-build failed, falling back to original RST files"
        # Fall back to original approach
        cat > "$API_REFS_FILE" << EOFAPI
API Reference
=============

This document contains the complete API reference for pyhanami.

*Last update*: $(date)

EOFAPI
        
        find "$TEMP_DIR" -name "*.rst" -not -name "modules.rst" | sort | while read -r rst_file; do
            module_name=$(basename "$rst_file" .rst)
            clean_name=$(echo "$module_name" | sed 's/.*\.//g')
            
            echo "" >> "$API_REFS_FILE"
            echo "$clean_name" >> "$API_REFS_FILE"
            echo "$(printf -- '-%.0s' $(seq 1 ${#clean_name}))" >> "$API_REFS_FILE"
            echo "" >> "$API_REFS_FILE"
            
            cat "$rst_file" >> "$API_REFS_FILE"
            echo "" >> "$API_REFS_FILE"
        done
    else
        # Create the consolidated API Reference RST file
        cat > "$API_REFS_FILE" << EOF
API Reference
=============

This document contains the complete API reference for pyhanami.

*Last update*: $(date)

EOF
    
        # Process each generated text file and convert back to RST format
        find "$TEMP_DIR/output" -name "*.txt" | sort | while read -r txt_file; do
            module_name=$(basename "$txt_file" .txt)
            
            if [ "$module_name" != "modules" ] && [[ "$module_name" != *"package"* ]]; then
                # Clean up module name - extract just the final part
                clean_name=$(echo "$module_name" | sed 's/.*\.//g' | sed 's/ module$//g')
                
                echo "" >> "$API_REFS_FILE"
                echo "$clean_name" >> "$API_REFS_FILE"
                echo "$(printf -- '-%.0s' $(seq 1 ${#clean_name}))" >> "$API_REFS_FILE"
                echo "" >> "$API_REFS_FILE"
                
                # Convert the text content to RST format
                sed 's/^/   /' "$txt_file" >> "$API_REFS_FILE"
                echo "" >> "$API_REFS_FILE"
            fi
        done
    fi
    
    # Cleanup
    rm -rf "$TEMP_DIR"
    
    log "API documentation generated successfully: $API_REFS_FILE"
    log "RST file with actual content is ready for Read the Docs integration"
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
