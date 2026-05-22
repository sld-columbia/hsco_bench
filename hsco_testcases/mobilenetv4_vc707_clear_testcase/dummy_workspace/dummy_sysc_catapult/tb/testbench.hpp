// Copyright (c) 2011-2026 Columbia University, System Level Design Group
// SPDX-License-Identifier: Apache-2.0

#ifndef __TESTBENCH_HPP__
#define __TESTBENCH_HPP__

#pragma once

#include <systemc.h>
#include "dummy_conf_info.hpp"
#include "dummy_specs.hpp"
#include "dummy_data_types.hpp"
#include "esp_dma_info_sysc.hpp"
#include "../inc/core/systems/esp_dma_controller.hpp"

#include "ac_int.h"
#include "ac_fixed.h"
#include "ac_float.h"
#include "ac_std_float.h"
#include "ac_math.h"

#define __round_mask(x, y) ((y)-1)
#define round_up(x, y)     ((((x)-1) | __round_mask(x, y)) + 1)

SC_MODULE(testbench)
{
    sc_in<bool> CCS_INIT_S1(clk);
    sc_in<bool> CCS_INIT_S1(rst_bar);

    Connections::Out<conf_info_t> CCS_INIT_S1(conf_info);
    Connections::Out<ac_int<DMA_WIDTH>> CCS_INIT_S1(dma_read_chnl);
    Connections::In<ac_int<DMA_WIDTH>> CCS_INIT_S1(dma_write_chnl);
    Connections::In<dma_info_t> CCS_INIT_S1(dma_read_ctrl);
    Connections::In<dma_info_t> CCS_INIT_S1(dma_write_ctrl);
    sc_in<bool> CCS_INIT_S1(acc_done);

    sc_signal<bool> acc_rst;

    // Shared memory buffer model
    ac_int<DMA_WIDTH> *mem;

    // DMA controller instace
    esp_dma_controller<DMA_WIDTH, MEM_SIZE> *dmac;

    // SC_CTOR(testbench) {
    SC_HAS_PROCESS(testbench);
    testbench(const sc_module_name &name) :
        sc_module(name), mem(new ac_int<DMA_WIDTH>[MEM_SIZE]),
        dmac(new esp_dma_controller<DMA_WIDTH, MEM_SIZE>("dma-controller", mem))
    {

        SC_THREAD(proc);
        sensitive << clk.pos();
        async_reset_signal_is(rst_bar, false);

        dmac->clk(clk);
        dmac->rst(rst_bar);
        dmac->dma_read_ctrl(dma_read_ctrl);
        dmac->dma_read_chnl(dma_read_chnl);
        dmac->dma_write_ctrl(dma_write_ctrl);
        dmac->dma_write_chnl(dma_write_chnl);
        dmac->acc_done(acc_done);
        dmac->acc_rst(acc_rst);

        /* <<--params-default-->> */
        rows = 128;
        cols = 1;
        loaded_cols = 128;

        A_size = rows * loaded_cols;
        B_size = loaded_cols * cols;
        O_size = rows * cols;
        A_size_adj = round_up(rows * loaded_cols, DMA_WORD_PER_BEAT);
        B_size_adj = round_up(loaded_cols * cols, DMA_WORD_PER_BEAT);
        O_size_adj = round_up(rows * cols, DMA_WORD_PER_BEAT);

    }

    void proc(void);
    void load_data_to_memory(float* data, int size, int address);
    void read_data_from_memory(float* data, int size, int address);
    void invoke_accelerator(conf_info_t config);
    void random_init_data(float* data, int size);

    void golden_operation(float* A, float* B, float* O, int rows, int cols, int loaded_cols);
    void accelerated_operation(float* A, float* B, float* O, int rows, int cols, int loaded_cols);
    void verify_correctness(float* out, float* golden_out, int size);



    uint32_t rows;
    uint32_t cols;
    uint32_t loaded_cols;



    uint32_t A_size_adj;
    uint32_t B_size_adj;
    uint32_t O_size_adj;
    uint32_t A_size;
    uint32_t B_size;
    uint32_t O_size;


    // {{data_pointers}}

    int err;
    
};

#endif
