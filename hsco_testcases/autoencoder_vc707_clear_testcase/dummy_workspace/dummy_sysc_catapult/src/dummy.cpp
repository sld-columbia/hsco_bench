// Copyright (c) 2011-2026 Columbia University, System Level Design Group
// SPDX-License-Identifier: Apache-2.0

#include "dummy.hpp"
#include <mc_scverify.h>

void dummy_sysc_catapult::config()
{
    conf_info.Reset();
    conf1.ResetWrite();
    conf2b.ResetWrite();
    conf3b.ResetWrite();

    sync01.ResetWrite();
    sync02b.ResetWrite();
    sync03b.ResetWrite();

    wait();

    while (1) {
        conf_info_t conf_di = conf_info.Pop();

        conf1.Push(conf_di);
        conf2b.Push(conf_di);
        conf3b.Push(conf_di);

        sync01.Push(1);
        sync02b.Push(1);
        sync03b.Push(1);
    }
}

void dummy_sysc_catapult::load()
{
    bool ping_pong = false;
    dma_read_chnl.Reset();
    dma_read_ctrl.Reset();

    sync01.ResetRead();
    sync12b.reset_sync_out();

    conf1.ResetRead();


    wait();

    while (1) {
        // wait for configuration
        sync01.Pop();

        bool ping       = true;

        conf_info_t conf = conf1.Pop();

        /* <<--local-params-->> */
        uint32_t rows = conf.rows;
        uint32_t cols = conf.cols;
        uint32_t loaded_cols = conf.loaded_cols;


        int A_length = rows * loaded_cols;
        uint32_t load_A_offset = 0;

        // Load snippet start
        for (int rem = A_length; rem > 0; rem -= A_PLM_IN_WORD) {
            uint32_t len = rem > A_PLM_IN_WORD ? A_PLM_IN_WORD : rem;
            dma_info_t dma_info(load_A_offset / DMA_WORD_PER_BEAT, len / DMA_WORD_PER_BEAT, DMA_SIZE, 0);
            load_A_offset += len;
            dma_read_ctrl.Push(dma_info);

            for (uint32_t i = 0; i < len; i += DMA_WORD_PER_BEAT) {
                DMA_WORD data_dma = dma_read_chnl.Pop();

                // Write to PLM (all DMA_WORD_PER_BEAT words in one cycle)
                for (uint32_t k = 0; k < DMA_WORD_PER_BEAT; k++) {
                    int2fx(data_dma.slc<DATA_WIDTH>(k * DATA_WIDTH), A_ping[0][i + k]);
                }
            }
        }
        // Load snippet end
        int B_length = loaded_cols * cols;
        uint32_t load_B_offset = round_up(A_length, DMA_WORD_PER_BEAT);

        // Load snippet start
        for (int rem = B_length; rem > 0; rem -= B_PLM_IN_WORD) {
            uint32_t len = rem > B_PLM_IN_WORD ? B_PLM_IN_WORD : rem;
            dma_info_t dma_info(load_B_offset / DMA_WORD_PER_BEAT, len / DMA_WORD_PER_BEAT, DMA_SIZE, 0);
            load_B_offset += len;
            dma_read_ctrl.Push(dma_info);

            for (uint32_t i = 0; i < len; i += DMA_WORD_PER_BEAT) {
                DMA_WORD data_dma = dma_read_chnl.Pop();

                // Write to PLM (all DMA_WORD_PER_BEAT words in one cycle)
                for (uint32_t k = 0; k < DMA_WORD_PER_BEAT; k++) {
                    int2fx(data_dma.slc<DATA_WIDTH>(k * DATA_WIDTH), B_ping[0][i + k]);
                }
            }
        }
        // Load snippet end

        
        sync12b.sync_out();
        

        ping_pong = !ping_pong;

        
    }
}

void dummy_sysc_catapult::compute()
{

    bool ping_pong     = false;
    bool out_ping_pong = false;
    sync12b.reset_sync_in();
    sync2b3b.reset_sync_out();

    sync02b.ResetRead();
    conf2b.ResetRead();


    wait();

    while (1) {

        // wait for configuration
        sync02b.Pop();

        conf_info_t conf = conf2b.Pop();

        /* <<--local-params-->> */
        uint32_t rows = conf.rows;
        uint32_t cols = conf.cols;
        uint32_t loaded_cols = conf.loaded_cols;


        #pragma hls_pipeline_init_interval 1
        for(int i = 0; i < O_PLM_OUT_WORD; i++){
            O_ping[0][i] = 0;
            wait();
        }


        
        sync12b.sync_in();
        wait();

        // Kernel snippet start
        //
        // IDENTITY KERNEL (STRUCTURAL DUMMY ONLY)
        //
        // This reference accelerator demonstrates the full ESP accelerator
        // pipeline (config -> load -> compute -> store), DMA interface, PLM
        // usage, register decoding, and software-driver handshake. The
        // compute phase intentionally does NO useful computation: it simply
        // copies the first (rows * cols) elements from A_ping into O_ping.
        // Input B is loaded but ignored. Replace this block with your own
        // kernel when you create your accelerator.
        //
        uint32_t O_size = rows * cols;
        for(uint32_t i = 0; i < O_size; i++){
            O_ping[0][i] = A_ping[0][i];
        }
        // Kernel snippet end

        sync2b3b.sync_out();

    }
}

void dummy_sysc_catapult::store()
{

    bool ping_pong = false;
    dma_write_chnl.Reset();
    dma_write_ctrl.Reset();

    sync2b3b.reset_sync_in();

    sync03b.ResetRead();
    conf3b.ResetRead();


    acc_done.write(false);

    wait();

    while (1) {

        sync03b.Pop();

        conf_info_t conf = conf3b.Pop();

        /* <<--local-params-->> */
        uint32_t rows = conf.rows;
        uint32_t cols = conf.cols;
        uint32_t loaded_cols = conf.loaded_cols;


        bool ping = true;

        int A_length = rows * loaded_cols;
        int B_length = loaded_cols * cols;
        uint32_t store_O_offset = round_up(A_length, DMA_WORD_PER_BEAT) + round_up(B_length, DMA_WORD_PER_BEAT);

        wait();

        uint32_t O_length = rows * cols;

        // Store snippet start
        for (int rem = O_length; rem > 0; rem -= O_PLM_OUT_WORD)
        {
            sync2b3b.sync_in();
            uint32_t len = rem > O_PLM_OUT_WORD ? O_PLM_OUT_WORD : rem;
            dma_info_t dma_info(store_O_offset / DMA_WORD_PER_BEAT, len / DMA_WORD_PER_BEAT, DMA_SIZE, 0);
            dma_write_ctrl.Push(dma_info);


            for (uint32_t i = 0; i < len; i += DMA_WORD_PER_BEAT)
            {
                DMA_WORD data_dma;

                // Read from PLM
                wait();
                for (uint32_t k = 0; k < DMA_WORD_PER_BEAT; k++)
                {
                    FPDATA_WORD tmp_data;
                    fx2int(O_ping[0][i + k], tmp_data);
                    data_dma.set_slc(k * DATA_WIDTH, tmp_data);

                }
                dma_write_chnl.Push(data_dma);
            }
        }
        // Store snippet end


        wait();

        acc_done.write(true);
        wait();
        acc_done.write(false);
    }
}
