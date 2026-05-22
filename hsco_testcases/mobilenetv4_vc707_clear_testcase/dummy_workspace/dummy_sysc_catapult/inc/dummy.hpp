// Copyright (c) 2011-2026 Columbia University, System Level Design Group
// SPDX-License-Identifier: Apache-2.0

#ifndef __TOP_HPP__
#define __TOP_HPP__

#pragma once

// #include "mem_wrap.hpp"
// #include "ac_shared_bank_array.h"
#include <ac_shared_bank_array.h>
#include "dummy_data_types.hpp"
#include "dummy_specs.hpp"
#include "dummy_conf_info.hpp"

#define __round_mask(x, y) ((y)-1)
#define round_up(x, y)     ((((x)-1) | __round_mask(x, y)) + 1)

SC_MODULE(dummy_sysc_catapult)
{
  public:
    sc_in<bool> CCS_INIT_S1(clk);
    sc_in<bool> CCS_INIT_S1(rst);
    sc_out<bool> CCS_INIT_S1(acc_done);

    Connections::In<conf_info_t> CCS_INIT_S1(conf_info);
    Connections::In<ac_int<DMA_WIDTH>> CCS_INIT_S1(dma_read_chnl);
    Connections::Out<ac_int<DMA_WIDTH>> CCS_INIT_S1(dma_write_chnl);
    Connections::Out<dma_info_t> CCS_INIT_S1(dma_read_ctrl);
    Connections::Out<dma_info_t> CCS_INIT_S1(dma_write_ctrl);

    void config(void);
    void load(void);
    void compute(void);
    void store(void);

    SC_CTOR(dummy_sysc_catapult) 
    {

        SC_THREAD(config);
        sensitive << clk.pos();
        async_reset_signal_is(rst, false);

        SC_THREAD(load);
        sensitive << clk.pos();
        async_reset_signal_is(rst, false);

        SC_THREAD(compute);
        sensitive << clk.pos();
        async_reset_signal_is(rst, false);

        SC_THREAD(store);
        sensitive << clk.pos();
        async_reset_signal_is(rst, false);
    }

    Connections::SyncChannel CCS_INIT_S1(sync12b);
    Connections::SyncChannel CCS_INIT_S1(sync2b3b);

    Connections::Combinational<bool> CCS_INIT_S1(sync01);
    Connections::Combinational<bool> CCS_INIT_S1(sync02b);
    Connections::Combinational<bool> CCS_INIT_S1(sync03b);

    Connections::Combinational<conf_info_t> CCS_INIT_S1(conf1);
    Connections::Combinational<conf_info_t> CCS_INIT_S1(conf2b);
    Connections::Combinational<conf_info_t> CCS_INIT_S1(conf3b);

    

    // ac_shared_bank_array_2D<FPDATA, inbks, inebks> A_ping;
    // ac_shared_bank_array_2D<FPDATA, inbks, inebks> A_pong;
    // ac_shared_bank_array_2D<FPDATA, inbks, inebks> B_ping;
    // ac_shared_bank_array_2D<FPDATA, inbks, inebks> B_pong;
    // ac_shared_bank_array_2D<FPDATA, outbks, outebks> O_ping;
    // ac_shared_bank_array_2D<FPDATA, outbks, outebks> O_pong;
    ac_shared_bank_array_2D<FPDATA, inbks, A_inebks> A_ping;
    ac_shared_bank_array_2D<FPDATA, inbks, B_inebks> B_ping;
    ac_shared_bank_array_2D<FPDATA, outbks, O_outebks> O_ping;


};

#endif
