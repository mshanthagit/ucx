#!/bin/bash
# Unified build script for ROCm Device PUT Client and Server (Warp-Level)
# Supports configurable UCX installation path

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default UCX installation path
DEFAULT_UCX_PATH="path-to-ucx-installation"

# Function to print usage
print_usage() {
    cat << EOF
Usage: $0 [UCX_PATH] [TARGET]

Arguments:
  UCX_PATH    Path to UCX installation (optional)
              Default: ${DEFAULT_UCX_PATH}
              Can also be set via UCX_INSTALL_PATH environment variable
  
  TARGET      What to build (optional)
              Options: client, server, both, all
              Default: both

Examples:
  $0                                    # Build both with default UCX path
  $0 /custom/ucx/path                   # Build both with custom UCX path
  $0 /custom/ucx/path client            # Build only client
  $0 server                             # Build only server with default path
  UCX_INSTALL_PATH=/custom/path $0      # Build both using env variable

EOF
}

# Function to build a target
build_target() {
    local target=$1
    local ucx_path=$2
    local source_file=""
    local output_file=""
    
    if [ "$target" = "client" ]; then
        source_file="rocm_device_put_client_warp.cpp"
        output_file="rocm_device_put_client_warp.exe"
    elif [ "$target" = "server" ]; then
        source_file="rocm_device_put_server_warp.cpp"
        output_file="rocm_device_put_server_warp.exe"
    else
        echo -e "${RED}Error: Invalid target '$target'${NC}"
        return 1
    fi
    
    echo -e "${YELLOW}Building $target...${NC}"
    
    if ! hipcc -DHAVE_ROCM=1 \
          -I"${ucx_path}/include" \
          -L"${ucx_path}/lib" \
          -Wl,-rpath,"${ucx_path}/lib" \
          -o "$output_file" \
          "$source_file" \
          -lucp -luct -lucs -lm; then
        echo -e "${RED}Failed to build $target${NC}"
        return 1
    fi
    
    echo -e "${GREEN}Successfully built $target: $output_file${NC}"
    return 0
}

# Parse arguments
UCX_PATH=""
TARGET="both"

# Check for help flag
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    print_usage
    exit 0
fi

# Parse first argument - could be UCX path or target
if [ -n "$1" ]; then
    # Check if first argument is a valid target
    if [ "$1" = "client" ] || [ "$1" = "server" ] || [ "$1" = "both" ] || [ "$1" = "all" ]; then
        TARGET="$1"
    else
        # Assume it's a UCX path
        UCX_PATH="$1"
        # Check for second argument as target
        if [ -n "$2" ]; then
            TARGET="$2"
        fi
    fi
fi

# Determine final UCX path (priority: arg > env var > default)
if [ -z "$UCX_PATH" ]; then
    if [ -n "$UCX_INSTALL_PATH" ]; then
        UCX_PATH="$UCX_INSTALL_PATH"
    else
        UCX_PATH="$DEFAULT_UCX_PATH"
    fi
fi

# Validate UCX installation path
if [ ! -d "$UCX_PATH" ]; then
    echo -e "${RED}Error: UCX installation path does not exist: $UCX_PATH${NC}"
    echo "Please provide a valid UCX installation path."
    echo ""
    print_usage
    exit 1
fi

if [ ! -d "$UCX_PATH/include" ] || [ ! -d "$UCX_PATH/lib" ]; then
    echo -e "${RED}Error: Invalid UCX installation at $UCX_PATH${NC}"
    echo "Missing include/ or lib/ directories."
    exit 1
fi

# Set LD_LIBRARY_PATH
export LD_LIBRARY_PATH="${UCX_PATH}/lib:$LD_LIBRARY_PATH"

echo "========================================="
echo "UCX Installation Path: $UCX_PATH"
echo "Build Target: $TARGET"
echo "========================================="
echo ""

# Build based on target
BUILD_FAILED=0

if [ "$TARGET" = "both" ] || [ "$TARGET" = "all" ]; then
    build_target "client" "$UCX_PATH" || BUILD_FAILED=1
    echo ""
    build_target "server" "$UCX_PATH" || BUILD_FAILED=1
elif [ "$TARGET" = "client" ]; then
    build_target "client" "$UCX_PATH" || BUILD_FAILED=1
elif [ "$TARGET" = "server" ]; then
    build_target "server" "$UCX_PATH" || BUILD_FAILED=1
else
    echo -e "${RED}Error: Invalid target '$TARGET'${NC}"
    echo "Valid targets: client, server, both, all"
    print_usage
    exit 1
fi

echo ""
if [ $BUILD_FAILED -eq 0 ]; then
    echo -e "${GREEN}=========================================${NC}"
    echo -e "${GREEN}Build completed successfully!${NC}"
    echo -e "${GREEN}=========================================${NC}"
    exit 0
else
    echo -e "${RED}=========================================${NC}"
    echo -e "${RED}Build failed!${NC}"
    echo -e "${RED}=========================================${NC}"
    exit 1
fi
