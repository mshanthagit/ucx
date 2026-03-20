/*
 * ROCm Device-Initiated PUT Client (Warp-Level)
 * 
 * Client component for device-initiated RDMA PUT operations.
 * Sends data to server via UCX device API from HIP kernel.
 * 
 * Usage: ./rocm_device_put_client_warp.exe [gpu_id]
 * Default: gpu_id=1
 */

#include <hip/hip_runtime.h>
#include <ucp/api/ucp.h>
#include <ucp/api/device/ucp_host.h>
#include <ucp/api/device/ucp_device_impl.h>

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

static int wait_flag(const char *name, int timeout_sec) {
    int count = 0;
    while (access(name, F_OK) != 0) {
        usleep(10000);  /* 10ms */
        if (++count > timeout_sec * 100) return -1;
    }
    return 0;
}

static size_t read_binary(const char *name, void *data, size_t max_len) {
    FILE *f = fopen(name, "rb");
    if (!f) return 0;
    size_t len = fread(data, 1, max_len, f);
    fclose(f);
    return len;
}

static uint64_t read_uint64(const char *name) {
    FILE *f = fopen(name, "r");
    if (!f) return 0;
    uint64_t val;
    fscanf(f, "%lu", &val);
    fclose(f);
    return val;
}

static size_t read_size(const char *name) {
    FILE *f = fopen(name, "r");
    if (!f) return 0;
    size_t val;
    fscanf(f, "%zu", &val);
    fclose(f);
    return val;
}

/*
 * HIP Kernel: Device-initiated PUT with wavefront-level parallelism
 */
__global__ void device_put_kernel_wavefront(
    ucp_device_local_mem_list_h src_mem_list_h,
    ucp_device_remote_mem_list_h dst_mem_list_h,
    size_t length,
    ucs_status_t *status_out)
{
    ucp_device_request_t req;
    ucs_status_t status;
    /* All threads in wavefront 0 cooperate */
    if (blockIdx.x == 0 && threadIdx.x < 64) {
        status = ucp_device_put<UCS_DEVICE_LEVEL_WARP>(
            src_mem_list_h,     /* source handle */
            0,                  /* src_mem_list_index */
            0,                  /* src_offset */
            dst_mem_list_h,     /* destination handle */
            0,                  /* dst_mem_list_index */
            0,                  /* dst_offset */
            length,             /* length in bytes */
            0,                  /* channel_id */
            UCP_DEVICE_FLAG_NODELAY,
            &req
        );
        
        if (status == UCS_INPROGRESS) {
            do {
                status = ucp_device_progress_req<UCS_DEVICE_LEVEL_WARP>(&req);
            } while (status == UCS_INPROGRESS);
        }
        
        /* Only thread 0 writes status */
        if (threadIdx.x == 0 && status_out != nullptr) {
            *status_out = status;
        }
    }
    
    __threadfence_system();
}

int main(int argc, char **argv) {
    int gpu_id = (argc > 1) ? atoi(argv[1]) : 0;  /* Use device 0 (HIP_VISIBLE_DEVICES controls which physical GPU) */
    ucp_context_h ctx;
    ucp_worker_h worker;
    ucp_mem_h memh = nullptr;
    ucp_ep_h ep = nullptr;
    ucp_rkey_h rkey = nullptr;
    ucp_device_local_mem_list_h src_mem_list_h = nullptr;
    ucp_device_remote_mem_list_h dst_mem_list_h = nullptr;
    int *d_buf = nullptr;
    int *h_buf = nullptr;
    ucs_status_t *d_status = nullptr;
    size_t sizes[] = {1*MB, 2*MB, 3*MB, 4*MB};  /* Test only 4MB for initial testing */
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    
    char addr_buf[512];
    char rkey_buf[256];
    size_t addr_len, rkey_len;
    uint64_t remote_addr;
    
    ucp_ep_params_t ep_params;
    ucp_mem_map_params_t mparams;
    ucp_request_param_t req_param = {};
    ucs_status_ptr_t flush_req;

    printf("=== ROCm Device PUT Client (Warp-Level) ===\n");
    printf("Using GPU %d\n\n", gpu_id);

    CHECK_HIP(hipSetDevice(gpu_id));

    /* Initialize UCX with RMA + DEVICE features */
    printf("Client: Initializing UCX...\n"); fflush(stdout);
    ucp_params_t params = {};
    params.field_mask = UCP_PARAM_FIELD_FEATURES;
    params.features   = UCP_FEATURE_RMA | UCP_FEATURE_AMO64 | UCP_FEATURE_DEVICE;
    CHECK_UCS(ucp_init(&params, NULL, &ctx));


    /* Allocate GPU memory */
    printf("Client: Allocating GPU memory...\n"); fflush(stdout);
    CHECK_HIP(hipMalloc(&d_buf, MAX_SIZE));
    CHECK_HIP(hipMalloc(&d_status, sizeof(ucs_status_t)));

    ucp_worker_params_t wparams = {};
    wparams.field_mask  = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    wparams.thread_mode = UCS_THREAD_MODE_SINGLE;
    CHECK_UCS(ucp_worker_create(ctx, &wparams, &worker));
    printf("Client: Worker created\n"); fflush(stdout);

    /* Wait for server */
    printf("Client: Waiting for server...\n"); fflush(stdout);
    if (wait_flag("ucx_server_ready.flag", TIMEOUT_SEC) < 0) {
        printf("Client: TIMEOUT waiting for server!\n");
        fflush(stdout);
        goto cleanup_early;
    }
    printf("Client: Server is ready\n"); fflush(stdout);

    /* Read server's address, rkey, and remote address */
    addr_len = read_binary("ucx_server_addr.bin", addr_buf, sizeof(addr_buf));
    rkey_len = read_binary("ucx_server_rkey.bin", rkey_buf, sizeof(rkey_buf));
    remote_addr = read_uint64("ucx_server_remote_addr.txt");
    printf("Client: Read server info (addr_len=%zu, rkey_len=%zu, remote_addr=0x%lx)\n",
           addr_len, rkey_len, remote_addr); fflush(stdout);

    /* Create endpoint to server */
    printf("Client: Creating endpoint...\n"); fflush(stdout);
    memset(&ep_params, 0, sizeof(ep_params));
    ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
    ep_params.address    = (ucp_address_t*)addr_buf;
    CHECK_UCS(ucp_ep_create(worker, &ep_params, &ep));
    printf("Client: Endpoint created\n"); fflush(stdout);

    /* Unpack server's rkey */
    printf("Client: Unpacking rkey...\n"); fflush(stdout);
    CHECK_UCS(ucp_ep_rkey_unpack(ep, rkey_buf, &rkey));
    printf("Client: Rkey unpacked\n"); fflush(stdout);

    /* Register GPU memory with UCX */
    memset(&mparams, 0, sizeof(mparams));
    mparams.field_mask  = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                          UCP_MEM_MAP_PARAM_FIELD_LENGTH |
                          UCP_MEM_MAP_PARAM_FIELD_MEMORY_TYPE;
    mparams.address     = d_buf;
    mparams.length      = MAX_SIZE;
    mparams.memory_type = UCS_MEMORY_TYPE_ROCM;
    
    ucp_worker_flush(worker);
    CHECK_UCS(ucp_mem_map(ctx, &mparams, &memh));
    printf("Client: GPU memory registered at %p\n", d_buf); fflush(stdout);

    /* Allocate host buffer for initialization */
    h_buf = (int*)malloc(MAX_SIZE);
    for (int i = 0; i < MAX_SIZE/sizeof(int); i++) {
        h_buf[i] = i;
    }
    printf( "src[9] ================================= %d\n", h_buf[9]);
    printf("Client: Host buffer initialized\n"); fflush(stdout);

    /* Test each size */
    for (int i = 0; i < num_sizes; i++) {
        size_t size = sizes[i];
        
        /* Wait for server to set test size */
        printf("Client: Waiting for test_size=%zu MB...\n", size/MB); fflush(stdout);
        int timeout_count = 0;
        size_t server_size;
        while ((server_size = read_size("ucx_test_size.txt")) != size) {
            usleep(10000);
            if (++timeout_count > TIMEOUT_SEC * 100) {
                printf("Client: TIMEOUT waiting for test_size!\n");
                fflush(stdout);
                goto cleanup;
            }
        }
        printf("Client: Test size matched\n"); fflush(stdout);
        
        /* Initialize GPU buffer with test pattern */
        CHECK_HIP(hipMemcpy(d_buf, h_buf, size, hipMemcpyHostToDevice));
        
        /* Release previous memory list handle if exists */
        if (dst_mem_list_h != nullptr) {
            ucp_device_mem_list_release(dst_mem_list_h);
            dst_mem_list_h = nullptr;
            src_mem_list_h = nullptr; 
        }
        
        /* Create single memory list handle with both local and remote info */
        printf("Client: Creating device memory list handle...\n"); fflush(stdout);
        ucp_device_mem_list_elem_t elem[1];
        elem[0].field_mask  = UCP_DEVICE_MEM_LIST_ELEM_FIELD_MEMH |
                              UCP_DEVICE_MEM_LIST_ELEM_FIELD_RKEY |
                              UCP_DEVICE_MEM_LIST_ELEM_FIELD_LOCAL_ADDR |
                              UCP_DEVICE_MEM_LIST_ELEM_FIELD_REMOTE_ADDR |
                              UCP_DEVICE_MEM_LIST_ELEM_FIELD_LENGTH |
                              UCP_DEVICE_MEM_LIST_ELEM_FIELD_EP;
        elem[0].memh        = memh;
        elem[0].rkey        = rkey;
        elem[0].local_addr  = d_buf;
        elem[0].remote_addr = remote_addr;
        elem[0].length      = size;
        elem[0].ep          = ep;

        ucp_device_mem_list_params_t list_params = {};
        list_params.field_mask   = UCP_DEVICE_MEM_LIST_PARAMS_FIELD_ELEMENTS |
                                   UCP_DEVICE_MEM_LIST_PARAMS_FIELD_NUM_ELEMENTS |
                                   UCP_DEVICE_MEM_LIST_PARAMS_FIELD_ELEMENT_SIZE |
                                   UCP_DEVICE_MEM_LIST_PARAMS_FIELD_WORKER;
        list_params.element_size = sizeof(ucp_device_mem_list_elem_t);
        list_params.num_elements = 1;
        list_params.elements     = elem;
        list_params.worker       = worker;

        /* Create memory list with retry on connection */
        ucs_status_t status;
        int retry_count = 0;
        const int MAX_RETRIES = 1000;  // ~10 seconds with 10ms sleep
        
        do {
            /* Progress worker multiple times to establish connection */
            for (int j = 0; j < 10; j++) {
                ucp_worker_progress(worker);
            }
            printf("Client: before device local mem list create\n");
            status = ucp_device_local_mem_list_create(&list_params, &src_mem_list_h);
            if (status != UCS_OK) {
                printf("Client: local mem list creation failed\n");
            }
            
            status = ucp_device_remote_mem_list_create(&list_params, &dst_mem_list_h);

            if (status == UCS_ERR_NOT_CONNECTED) {
                usleep(10000);  // 10ms
                retry_count++;
                if (retry_count >= MAX_RETRIES) {
                    printf("Client: TIMEOUT creating device memory list (still not connected after %d retries)\n", retry_count);
                    fflush(stdout);
                    goto cleanup;
                }
                if (retry_count % 100 == 0) {
                    printf("Client: Still waiting for connection... (retry %d/%d)\n", retry_count, MAX_RETRIES);
                    fflush(stdout);
                }
            }
        } while (status == UCS_ERR_NOT_CONNECTED);
        
        if (status == UCS_ERR_NO_DEVICE) {
            printf("Client: No device lanes available, skipping device PUT\n");
            fflush(stdout);
            goto cleanup;
        }
        
        if (status != UCS_OK) {
            printf("Client: Failed to create device memory list: %s\n", ucs_status_string(status));
            fflush(stdout);
            goto cleanup;
        }
        
//        src_mem_list_h = reinterpret_cast<ucp_device_local_mem_list_h>(dst_mem_list_h);
        printf("Client: Memory list handle created: %p (used as both src and dst)\n", dst_mem_list_h);
        fflush(stdout);

        /* Ensure GPU can see the handle data */
        CHECK_HIP(hipDeviceSynchronize());

        /* Initialize status on device */
        ucs_status_t init_status = UCS_ERR_NOT_IMPLEMENTED;
        CHECK_HIP(hipMemcpy(d_status, &init_status, sizeof(ucs_status_t), 
                            hipMemcpyHostToDevice));

        /* Launch HIP kernel to perform device-initiated PUT with warp-level parallelism */
        printf("Client: Launching warp-level device PUT kernel for %zu MB...\n", size/MB);
        fflush(stdout);
        
        hipLaunchKernelGGL(device_put_kernel_wavefront, 
                           dim3(1),      // 1 block
                           dim3(64),     // 64 threads (wavefront size)
                           0,            // shared memory
                           0,            // stream 
                           src_mem_list_h, dst_mem_list_h, size, d_status);
        
        /* Wait for kernel completion */
        CHECK_HIP(hipDeviceSynchronize());
        
        /* Check status from device */
        ucs_status_t kernel_status;
        CHECK_HIP(hipMemcpy(&kernel_status, d_status, sizeof(ucs_status_t), 
                            hipMemcpyDeviceToHost));
        
        if (kernel_status != UCS_OK) {
            printf("Client: Device PUT failed with status: %s\n", 
                   ucs_status_string(kernel_status));
            fflush(stdout);
            goto cleanup;
        }
        printf("Client: Warp-level device PUT completed successfully\n"); fflush(stdout);
        
        /* Flush endpoint to ensure all operations complete */
        printf("Client: Flushing endpoint...\n"); fflush(stdout);
        ucp_request_param_t flush_param = {};
        ucs_status_ptr_t req = ucp_ep_flush_nbx(ep, &flush_param);
        if (UCS_PTR_IS_PTR(req)) {
            while (ucp_request_check_status(req) == UCS_INPROGRESS) {
                ucp_worker_progress(worker);
            }
            ucp_request_free(req);
        }
        printf("Client: Flush completed\n"); fflush(stdout);
        
        /* Signal done */
        write_flag("ucx_client_done.flag");
        printf("Client: Signaled done\n"); fflush(stdout);
        
        /* Wait for server to clear flag before next test */
        timeout_count = 0;
        while (access("ucx_client_done.flag", F_OK) == 0) {
            usleep(10000);
            if (++timeout_count > TIMEOUT_SEC * 100) {
                printf("Client: TIMEOUT waiting for server to clear flag!\n");
                fflush(stdout);
                goto cleanup;
            }
        }
        printf("Client: Server cleared done flag\n"); fflush(stdout);
    }

cleanup:
    /* Cleanup */
    free(h_buf);
    if (dst_mem_list_h != nullptr) {
        ucp_device_mem_list_release(dst_mem_list_h);
        /* src_mem_list_h points to same handle, don't double-free */
    }
    ucp_rkey_destroy(rkey);
    ucp_ep_destroy(ep);
    CHECK_HIP(hipFree(d_status));
    CHECK_HIP(hipFree(d_buf));
    ucp_mem_unmap(ctx, memh);
cleanup_early:
    ucp_worker_destroy(worker);
    ucp_cleanup(ctx);
    printf("\nClient: Complete\n");
    return 0;
}
