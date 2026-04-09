/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2018. ALL RIGHTS RESERVED.
 * Copyright (C) Advanced Micro Devices, Inc. 2019. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCT_ROCM_IPC_CACHE_H_
#define UCT_ROCM_IPC_CACHE_H_

#include <ucs/datastruct/pgtable.h>
#include <ucs/datastruct/list.h>
#include <ucs/datastruct/khash.h>
#include <ucs/type/spinlock.h>
#include "rocm_ipc_md.h"


/* Forward declaration */
typedef struct uct_rocm_ipc_cache uct_rocm_ipc_cache_t;


/**
 * Peer identification key for cache hash table
 */
typedef struct uct_rocm_ipc_peer_key {
    pid_t        pid;
    ucs_sys_ns_t pid_ns;
} uct_rocm_ipc_peer_key_t;


static UCS_F_ALWAYS_INLINE int
uct_rocm_ipc_peer_key_equal(uct_rocm_ipc_peer_key_t key1,
                            uct_rocm_ipc_peer_key_t key2)
{
    return (key1.pid == key2.pid) && (key1.pid_ns == key2.pid_ns);
}


static UCS_F_ALWAYS_INLINE khint32_t
uct_rocm_ipc_peer_key_hash(uct_rocm_ipc_peer_key_t key)
{
    return kh_int64_hash_func((((uint64_t)key.pid) << 32) |
                              (key.pid_ns & 0xFFFFFFFFUL));
}


KHASH_INIT(rocm_ipc_peer_cache, uct_rocm_ipc_peer_key_t,
           uct_rocm_ipc_cache_t*, 1, uct_rocm_ipc_peer_key_hash,
           uct_rocm_ipc_peer_key_equal);


typedef struct uct_cuda_ipc_cache_region {
    ucs_pgt_region_t        super;        /**< Base class - page table region */
    ucs_list_link_t         list;         /**< List element */
    uct_rocm_ipc_key_t      key;          /**< Remote memory key */
    void                    *mapped_addr; /**< Local mapped address */
} uct_rocm_ipc_cache_region_t;

typedef struct uct_rocm_ipc_cache {
    pthread_rwlock_t      lock;       /**< Protects the page table */
    ucs_pgtable_t         pgtable;    /**< Page table to hold the regions */
    char                  *name;      /**< Cache name */
} uct_rocm_ipc_cache_t;


/**
 * Global cache manager - manages per-peer caches
 */
typedef struct uct_rocm_ipc_cache_manager {
    khash_t(rocm_ipc_peer_cache) hash;        /**< Hash table: peer_key -> cache */
    ucs_recursive_spinlock_t     lock;        /**< Protects hash table */
    unsigned long                max_regions; /**< Global max regions limit */
    size_t                       max_size;    /**< Global max size limit */
} uct_rocm_ipc_cache_manager_t;

ucs_status_t uct_rocm_ipc_create_cache(uct_rocm_ipc_cache_t **cache,
                                       const char *name);

void uct_rocm_ipc_destroy_cache(uct_rocm_ipc_cache_t *cache);

ucs_status_t uct_rocm_ipc_cache_map_memhandle(void *arg, uct_rocm_ipc_key_t *key,
                                              void **mapped_addr);

uct_rocm_ipc_cache_manager_t* uct_rocm_ipc_get_cache_manager(void);

#endif
