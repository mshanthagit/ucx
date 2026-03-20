#!/bin/bash
# Helper script to run ROCm Device PUT test with separate client/server

# Set LD_LIBRARY_PATH for UCX

# Clean up old coordination files
echo "Cleaning up old coordination files..."
rm -f ucx_*.bin ucx_*.txt ucx_*.flag

# Check if executables exist
if [ ! -f rocm_device_put_server_warp.exe ]; then
    echo "Error: Server executable not found. Run ./build_server_warp.sh first"
    exit 1
fi

if [ ! -f rocm_device_put_client_warp.exe ]; then
    echo "Error: Client executable not found. Run ./build_client_warp.sh first"
    exit 1
fi

# Parse GPU IDs (default: server=0, client=1)
SERVER_GPU=${1:-0}
CLIENT_GPU=${2:-1}

echo "========================================="
echo "ROCm Device PUT Test (Warp-Level)"
echo "Server GPU: $SERVER_GPU"
echo "Client GPU: $CLIENT_GPU"
echo "========================================="
echo ""

# Start server in background
echo "Starting server on GPU $SERVER_GPU..."
UCX_LOG_LEVEL=data UCX_LOG_FILE=server.log HIP_VISIBLE_DEVICES=$SERVER_GPU ./rocm_device_put_server_warp.exe 0 &
SERVER_PID=$!

# Give server time to initialize
sleep 2

# Start client
echo "Starting client on GPU $CLIENT_GPU..."
UCX_LOG_LEVEL=req UCX_LOG_FILE=client.log HIP_VISIBLE_DEVICES=$CLIENT_GPU ./rocm_device_put_client_warp.exe 0
CLIENT_EXIT=$?

# Wait for server to complete
wait $SERVER_PID
SERVER_EXIT=$?

# Clean up coordination files
echo ""
echo "Cleaning up coordination files..."
rm -f ucx_*.bin ucx_*.txt ucx_*.flag
