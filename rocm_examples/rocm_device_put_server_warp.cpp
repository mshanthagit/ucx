/*
 * ROCm Device-Initiated PUT Server (Warp-Level)
 * 
 * Server component for device-initiated PUT operations.
 * Receives data from client via UCX device API.
 * 
 * Usage: ./rocm_device_put_server_warp.exe [gpu_id]
 * Default: gpu_id=0
 */

#include <hip/hip_runtime.h>
#include <ucp/api/ucp.h>
#include <ucp/api/device/ucp_host.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>

#define MB (1024 * 1024)
#define MAX_SIZE (16 * MB)
#define TIMEOUT_SEC 30

#define CHECK_UCS(expr) do { \
    ucs_status_t _s = (expr); \
    if (_s != UCS_OK) { \
        fprintf(stderr, "UCX error at %s:%d: %s\n", __FILE__, __LINE__, \
                ucs_status_string(_s)); \
        exit(1); \
    } \
} while(0)

#define CHECK_HIP(expr) do { \
    hipError_t _e = (expr); \
    if (_e != hipSuccess) { \
        fprintf(stderr, "HIP error at %s:%d: %d %s\n", __FILE__, __LINE__, \
                _e, hipGetErrorString(_e)); \
        exit(1); \
    } \
} while(0)

/* File I/O helpers */
static void write_flag(const char *name) {
    FILE *f = fopen(name, "w");
    if (f) fclose(f);
}

static void remove_flag(const char *name) {
    unlink(name);
}

static int wait_flag(const char *name, int timeout_sec) {
    int count = 0;
    while (access(name, F_OK) != 0) {
        usleep(10000);  /* 10ms */
        if (++count > timeout_sec * 100) return -1;
    }
    return 0;
}

static void write_binary(const char *name, const void *data, size_t len) {
    FILE *f = fopen(name, "wb");
    if (f) {
        fwrite(data, 1, len, f);
        fclose(f);
    }
}

static void write_uint64(const char *name, uint64_t val) {
    FILE *f = fopen(name, "w");
    if (f) {
        fprintf(f, "%lu", val);
        fclose(f);
    }
}

static void write_size(const char *name, size_t val) {
    FILE *f = fopen(name, "w");
    if (f) {
        fprintf(f, "%zu", val);
        fclose(f);
    }
}

static void cleanup_files() {
    unlink("ucx_server_addr.bin");
    unlink("ucx_server_rkey.bin");
    unlink("ucx_server_remote_addr.txt");
    unlink("ucx_server_ready.flag");
    unlink("ucx_test_size.txt");
    unlink("ucx_client_done.flag");
}

static volatile int g_exit_flag = 0;

static void signal_handler(int sig) {
    g_exit_flag = 1;
    cleanup_files();
    exit(1);
}

int main(int argc, char **argv) {
    int gpu_id = (argc > 1) ? atoi(argv[1]) : 0;
    ucp_context_h ctx;
    ucp_worker_h worker;
    ucp_mem_h memh;
    void *rkey_buf;
    int *d_buf, *h_buf;
    size_t sizes[] = {1*MB, 2*MB, 3*MB, 4*MB};  /* Test only 4MB for initial testing */
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    size_t addr_len, rkey_len;

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    printf("=== ROCm Device PUT Server (Warp-Level) ===\n");
    printf("Using GPU %d\n\n", gpu_id);

    CHECK_HIP(hipSetDevice(gpu_id));


    /* Initialize UCX with RMA + DEVICE features */
    printf("Server: Initializing UCX...\n"); fflush(stdout);
    ucp_params_t params = {};
    params.field_mask = UCP_PARAM_FIELD_FEATURES;
    params.features   = UCP_FEATURE_RMA | UCP_FEATURE_AMO64 | UCP_FEATURE_DEVICE;
    CHECK_UCS(ucp_init(&params, NULL, &ctx));

    /* Allocate GPU memory */
    printf("Server: Allocating GPU memory...\n"); fflush(stdout);
    CHECK_HIP(hipMalloc(&d_buf, MAX_SIZE));    


    ucp_worker_params_t wparams = {};
    wparams.field_mask  = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    wparams.thread_mode = UCS_THREAD_MODE_SINGLE;
    CHECK_UCS(ucp_worker_create(ctx, &wparams, &worker));
    printf("Server: Worker created\n"); fflush(stdout);

    /* Get worker address */
    ucp_address_t *addr;
    CHECK_UCS(ucp_worker_get_address(worker, &addr, &addr_len));
    write_binary("ucx_server_addr.bin", addr, addr_len);
    ucp_worker_release_address(worker, addr);
    printf("Server: Worker address exported (len=%zu)\n", addr_len); fflush(stdout);

    /* Register GPU memory with UCX */
    ucp_mem_map_params_t mparams = {};
    mparams.field_mask  = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                          UCP_MEM_MAP_PARAM_FIELD_LENGTH |
                          UCP_MEM_MAP_PARAM_FIELD_MEMORY_TYPE;
    mparams.address     = d_buf;
    mparams.length      = MAX_SIZE;
    mparams.memory_type = UCS_MEMORY_TYPE_ROCM;
    ucp_worker_flush(worker);
    CHECK_UCS(ucp_mem_map(ctx, &mparams, &memh));
    printf("Server: GPU memory registered at %p\n", d_buf); fflush(stdout);

    /* Pack rkey and share */
    printf("Server: Packing rkey...\n"); fflush(stdout);
    CHECK_UCS(ucp_rkey_pack(ctx, memh, &rkey_buf, &rkey_len));
    write_binary("ucx_server_rkey.bin", rkey_buf, rkey_len);
    ucp_rkey_buffer_release(rkey_buf);
    write_uint64("ucx_server_remote_addr.txt", (uint64_t)d_buf);
    printf("Server: Rkey packed (len=%zu), remote_addr=0x%lx\n", 
           rkey_len, (uint64_t)d_buf); fflush(stdout);

    /* Signal ready */
    write_flag("ucx_server_ready.flag");
    printf("Server: Ready, waiting for client...\n"); fflush(stdout);

    /* Allocate host buffer for verification */
    h_buf = (int*)malloc(MAX_SIZE);

    /* Test each size */
    for (int i = 0; i < num_sizes; i++) {
        size_t size = sizes[i];
        
        /* Clear GPU buffer */
        CHECK_HIP(hipMemset(d_buf, 1, size));   // testing with a non-zero value
printf("Server: h_buf[10] = %d, d_buf[10] = %d\n", h_buf[10], d_buf[10]);
        /* Signal test size and wait for client */
        write_size("ucx_test_size.txt", size);
        printf("Server: Test size set to %zu MB, waiting for client...\n", 
               size/MB); fflush(stdout);
        
        if (wait_flag("ucx_client_done.flag", TIMEOUT_SEC) < 0) {
            printf("Server: TIMEOUT waiting for client!\n");
            fflush(stdout);
            break;
        }
        
        /* Progress worker while waiting */
        for (int j = 0; j < 10; j++) {
            ucp_worker_progress(worker);
            usleep(1000);
        }
        
        remove_flag("ucx_client_done.flag");
        
        /* Verify data */
        CHECK_HIP(hipMemcpy(h_buf, d_buf, size, hipMemcpyDeviceToHost));
        int errors = 0;
        int count = size / sizeof(int);
        printf("Server: h_buf[10] = %d, d_buf[10] = %d\n", h_buf[10], d_buf[10]);
//        h_buf[10] = -1;
        for (int j = 0; j < count && errors < 5; j++) {
            if (h_buf[j] != j) {
                printf("  Error at [%d]: expected %d, got %d\n", j, j, h_buf[j]);
                errors++;
            }
        }
        printf("Device PUT (Warp-Level) %2zu MB: %s\n", size/MB, errors ? "FAILED" : "OK");
        fflush(stdout);
    }

    /* Cleanup */
    free(h_buf);
    CHECK_HIP(hipFree(d_buf));
    ucp_mem_unmap(ctx, memh);
    ucp_worker_destroy(worker);
    ucp_cleanup(ctx);
    cleanup_files();
    printf("\nServer: Complete\n");
    return 0;
}
