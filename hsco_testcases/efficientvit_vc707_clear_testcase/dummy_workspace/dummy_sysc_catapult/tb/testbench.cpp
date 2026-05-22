// Copyright (c) 2011-2026 Columbia University, System Level Design Group
// SPDX-License-Identifier: Apache-2.0

#include "testbench.hpp"
#include "ac_math/ac_random.h"
#include <mc_connections.h>
#include <mc_scverify.h>


void testbench::random_init_data(float* data, int size)
{
    for(int i = 0; i < size; i++)
    {
        // Generate uniform random numbers in [-1, 1] with mean 0
        data[i] = 2.0f * static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 1.0f;
    }
}


void testbench::verify_correctness(float* out, float* golden_out, int size)
{
    for(int i = 0; i < size; i++)
    {
        if(out[i] != golden_out[i] && abs(out[i] - golden_out[i]) > 0.01)
        {
            cerr << "Error: out[" << i << "] = " << out[i] << ", golden_out[" << i << "] = " << golden_out[i] << endl;
            err++;
            cerr << "err:" << err << endl;
        }
        // cerr << "out[" << i << "] = " << out[i] << ", golden_out[" << i << "] = " << golden_out[i] << endl;
    }
    
}

void testbench::load_data_to_memory(float* data, int size, int address)
{
    for (uint32_t i = 0; i < (uint32_t)(size / DMA_WORD_PER_BEAT); i++)  {
        ac_int<DMA_WIDTH> data_bv;
        for (int j = 0; j < DMA_WORD_PER_BEAT; j++){
            ac_ieee_float32 data_fp = data[i * DMA_WORD_PER_BEAT + j];
            FPDATA tmp_fp = data_fp.convert_to_ac_fixed<FPDATA_WL,FPDATA_IL,true,AC_TRN, AC_WRAP>();
            // cerr << "tmp_fp = " << tmp_fp.to_double() << endl;
            FPDATA_WORD tmp_data1;
            fx2int(tmp_fp, tmp_data1);
            data_bv.set_slc(j * DATA_WIDTH, tmp_data1);  
        }
        mem[address + i] = data_bv;
    }
}

void testbench::read_data_from_memory(float* data, int size, int address)
{
    for (uint32_t i = 0; i < (uint32_t)(size / DMA_WORD_PER_BEAT); i++){
        ac_int<DATA_WIDTH> data_bv;
        for (int j = 0; j < DMA_WORD_PER_BEAT; j++){
            ac_int<DATA_WIDTH, false> tmp_data_int;
            tmp_data_int = mem[address + i].slc<DATA_WIDTH>(j * DATA_WIDTH);
            FPDATA tmp_fp;
            int2fx(tmp_data_int, tmp_fp);
            data[i * DMA_WORD_PER_BEAT + j] = (float) tmp_fp.to_double();
            // fprintf(stderr, "final data[%d] = %f\n", i * DMA_WORD_PER_BEAT + j, tmp_fp.to_double());
            // fprintf(stderr, "data[%d] = %f\n", i * DMA_WORD_PER_BEAT + j, data[i * DMA_WORD_PER_BEAT + j]);
        }
    }
}

void testbench::invoke_accelerator(conf_info_t config)
{
    // call the accelerator
    conf_info.Push(config);

    do {
        wait();
    } while (!acc_done.read());
}

void testbench::golden_operation(float* A, float* B, float* O_golden, int rows, int cols, int loaded_cols)
{
    // Golden reference for the IDENTITY kernel: copy the first (rows * cols)
    // elements of A into O_golden. B is intentionally unused, mirroring the
    // HLS kernel in src/dummy.cpp. Replace this with the true reference
    // computation when you adapt the accelerator for your own kernel.
    (void)B;
    (void)loaded_cols;
    uint32_t O_size_local = rows * cols;
    for(uint32_t i = 0; i < O_size_local; i++){
        O_golden[i] = A[i];
    }
}

void testbench::accelerated_operation(float* A, float* B, float* O, int rows, int cols, int loaded_cols)
{
    // here I will use the accelerator and verify the correctness
    // call matmul to calculate the golden solution
    // float* golden_out = (float*) calloc({{output_size}}, sizeof(float));
    float* O_golden = (float*) calloc(O_size, sizeof(float));

    golden_operation(A, B, O_golden, rows, cols, loaded_cols);

    // acccelerator flow
    // A. load data to memory
    load_data_to_memory(A, A_size, 0);
    load_data_to_memory(B, B_size, A_size_adj / DMA_WORD_PER_BEAT);

    // load_data_to_memory(A, A_row * A_col, 0);
    // load_data_to_memory(B, A_col * B_col, (A_row * A_col) / DMA_WORD_PER_BEAT);

    // B. call the accelerator
    conf_info_t config;
    // config.rows = A_row;
    // config.cols = B_col;
    // config.loaded_cols = A_col;
    config.rows = rows;
    config.cols = cols;
    config.loaded_cols = loaded_cols;


    invoke_accelerator(config);

    // C. read data from memory
    // read_data_from_memory(out, A_row * B_col, (A_row * A_col) / DMA_WORD_PER_BEAT + (B_col * A_col) / DMA_WORD_PER_BEAT);
    read_data_from_memory(O, O_size, A_size_adj / DMA_WORD_PER_BEAT + B_size_adj / DMA_WORD_PER_BEAT);


    // verify the correctness
    // verify_correctness(out, golden_out, A_row * B_col);
    verify_correctness(O, O_golden, O_size);


}


void testbench::proc()
{
    conf_info.Reset();

    wait();



    CCS_LOG("--------------------------------");
    CCS_LOG("ESP - dummy [Catapult HLS SystemC]");
    CCS_LOG("--------------------------------");

    int vocab_size=10, embed_dim=4, seq_len=3, ff_hidden=4;
    err = 0;
    
    // -------------------------
    // random init input
    // -------------------------

    float A[A_size];
    float B[B_size];
    float O[O_size];
    random_init_data(A, A_size);
    random_init_data(B, B_size);
    random_init_data(O, O_size);




    // -------------------------
    // Self-Attention Q,K,V
    // -------------------------
    
    accelerated_operation(A, B, O, rows, cols, loaded_cols);

    if (err > 0) CCS_LOG("SIMULATION FAILED ! \n - errors " << err);
    else
        CCS_LOG("SIMULATION PASSED ! ");

    sc_stop();

    wait();
}
