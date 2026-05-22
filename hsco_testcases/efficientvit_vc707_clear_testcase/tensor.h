/*
 * EfficientViT-M0 FPGA Baremetal Implementation
 * Tensor and model loading utilities (memory-mapped version)
 */

#ifndef TENSOR_H
#define TENSOR_H

#include <stdio.h>
#ifndef __riscv
#include <stdlib.h>
#endif
#include <string.h>
#include <stdint.h>
#include <math.h>

#ifdef __riscv
#define MB_ADDR 0xB0000000
#else
#define MB_ADDR 0x50000000
#endif

#define MAGIC 0x45564954  /* "EVIT" for EfficientViT */
#define VERSION 1
#define MAX_DIMS 4
#define MAX_NAME_LEN 256
#define MAX_TENSORS 512

typedef struct {
    char name[MAX_NAME_LEN];
    int ndim;
    int shape[MAX_DIMS];
    int size;  /* total number of elements */
    float *data;
} Tensor;

typedef struct {
    int num_tensors;
    Tensor tensors[MAX_TENSORS];
} Model;

static inline int tensor_size(const Tensor *t) {
    int size = 1;
    for (int i = 0; i < t->ndim; i++) {
        size *= t->shape[i];
    }
    return size;
}

static Tensor* find_tensor(Model *model, const char *name) {
    for (int i = 0; i < model->num_tensors; i++) {
        if (strcmp(model->tensors[i].name, name) == 0) {
            return &model->tensors[i];
        }
    }
    return NULL;
}

/* Helper function to read uint32_t from potentially unaligned address */
/* RISC-V may trap on unaligned 32-bit access, so read byte-by-byte */
static inline uint32_t read_u32_unaligned(const uint8_t *ptr) {
    return (uint32_t)ptr[0] | 
           ((uint32_t)ptr[1] << 8) | 
           ((uint32_t)ptr[2] << 16) | 
           ((uint32_t)ptr[3] << 24);
}

/* Load model from memory-mapped binary file (for FPGA baremetal) */
/* Binary format: */
/*   - magic (uint32_t): 0x45564954 ("EVIT") */
/*   - version (uint32_t): 1 */
/*   - num_tensors (uint32_t) */
/*   - For each tensor: */
/*     - name_len (uint32_t) */
/*     - name (char[name_len]) */
/*     - ndim (uint32_t) */
/*     - shape (uint32_t[ndim]) */
/*     - padding (0-3 bytes to align data to 4 bytes) */
/*     - data (float[product(shape)]) */
static int load_model_from_memory(Model *model) {
    printf("  Starting model load...\n");
    
    uint8_t *mem = (uint8_t *)MB_ADDR;
    
    uint32_t magic = read_u32_unaligned(mem + 0);
    uint32_t version = read_u32_unaligned(mem + 4);
    uint32_t num_tensors = read_u32_unaligned(mem + 8);
    
    printf("  Magic=0x%x, Ver=%u, Tensors=%u\n", 
           (unsigned)magic, (unsigned)version, (unsigned)num_tensors);
    
    /* Validate header */
    if (magic != MAGIC) {
        printf("Error: Invalid magic (expected 0x%x, got 0x%x)\n",
               (unsigned)MAGIC, (unsigned)magic);
        return -1;
    }
    
    if (version != VERSION || num_tensors > MAX_TENSORS) {
        printf("Error: Bad version or too many tensors\n");
        return -1;
    }
    
    model->num_tensors = num_tensors;
    
    /* Start reading tensors after header (3 uint32_t = 12 bytes) */
    size_t offset = 12;
    
    for (uint32_t i = 0; i < num_tensors; i++) {
        Tensor *t = &model->tensors[i];
        
        /* Read name length (use unaligned read) */
        uint32_t name_len = read_u32_unaligned(mem + offset);
        offset += 4;
        
        if (name_len >= MAX_NAME_LEN || name_len == 0) {
            printf("Error: T%d bad name_len=%d\n", (int)i, (int)name_len);
            return -1;
        }
        
        /* Copy name byte by byte */
        for (uint32_t k = 0; k < name_len; k++) {
            t->name[k] = mem[offset + k];
        }
        t->name[name_len] = '\0';
        offset += name_len;
        
        /* Read ndim (use unaligned read) */
        uint32_t ndim = read_u32_unaligned(mem + offset);
        offset += 4;
        t->ndim = ndim;
        
        if (ndim > MAX_DIMS) {
            printf("Error: T%d bad ndim=%d\n", (int)i, (int)ndim);
            return -1;
        }
        
        /* Read shape and calculate size (use unaligned read) */
        t->size = 1;
        for (uint32_t j = 0; j < ndim; j++) {
            uint32_t dim = read_u32_unaligned(mem + offset);
            offset += 4;
            t->shape[j] = dim;
            t->size *= dim;
        }
        
        /* Skip padding to align data to 4 bytes */
        size_t padding = (4 - (offset % 4)) % 4;
        offset += padding;
        
        /* Point to data (already in memory, no copy needed) */
        t->data = (float *)(mem + offset);
        offset += t->size * sizeof(float);
        
        /* Progress indicator */
        if (i % 20 == 0) printf(".");
    }
    printf(" Done! (%d tensors)\n", (int)num_tensors);
    
    return 0;
}

/* Free model memory (no-op for memory-mapped version) */
static void free_model(Model *model) {
    model->num_tensors = 0;
}

/* Allocate tensor using aligned_malloc for FPGA */
static float* alloc_tensor(int c, int h, int w) {
    int size = c * h * w;
    float *ptr = (float*)aligned_malloc(size * sizeof(float));
    if (ptr) {
        memset(ptr, 0, size * sizeof(float));
    }
    return ptr;
}

static float* alloc_tensor_4d(int n, int c, int h, int w) {
    int size = n * c * h * w;
    float *ptr = (float*)aligned_malloc(size * sizeof(float));
    if (ptr) {
        memset(ptr, 0, size * sizeof(float));
    }
    return ptr;
}

static float* alloc_tensor_1d(int size) {
    float *ptr = (float*)aligned_malloc(size * sizeof(float));
    if (ptr) {
        memset(ptr, 0, size * sizeof(float));
    }
    return ptr;
}

/* Print tensor info */
static void print_tensor_info(const Tensor *t) {
    printf("  %s: [", t->name);
    for (int i = 0; i < t->ndim; i++) {
        printf("%d", t->shape[i]);
        if (i < t->ndim - 1) printf(", ");
    }
    printf("] (%d elements)\n", t->size);
}

#endif /* TENSOR_H */
