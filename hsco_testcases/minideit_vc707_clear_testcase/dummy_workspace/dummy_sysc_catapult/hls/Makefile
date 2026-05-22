#Copyright (c) 2011-2026 Columbia University, System Level Design Group
#SPDX-License-Identifier: Apache-2.0

CXXFLAGS += -g -std=c++11 -Wall -Wno-unknown-pragmas -Wno-unused-variable -Wno-unused-label -Wno-maybe-uninitialized

ifndef SYSTEMC_HOME
$(error - Environment variable SYSTEMC_HOME must be defined)
endif
ifndef CONNECTIONS_HOME
$(error - Environment variable CONNECTIONS_HOME must be defined)
endif
ifndef MATCHLIB_HOME
$(error - Environment variable MATCHLIB_HOME must be defined)
endif
ifndef BOOST_HOME
$(error - Environment variable BOOST_HOME must be defined)
endif
ifndef RAPIDJSON_HOME
$(error - Environment variable RAPIDJSON_HOME must be defined)
endif
ifndef AC_TYPES
$(error - Environment variable AC_TYPES must be defined)
endif
ifndef AC_SIMUTILS
$(error - Environment variable AC_SIMUTILS must be defined)
endif


##### this part is required by the behavior simulation #####
ifeq ("$(CATAPULT_HOME)", "")
$(error please define CATAPULT_HOME required for Catapult HLS library headers)
endif

ifeq ("$(SYSTEMC)", "")
$(error please define SYSTEMC to execute a standalone simulation)
endif

DMA_WIDTH ?= 64
ifeq ("$(DMA_WIDTH)", "")
$(error DMA_WIDTH variable is not set! Please enter DMA_WIDTH=64 for Ariane or DMA_WIDTH=32 for Leon3 and Ibex before the corresponding make target)
endif

# Default to the compiler installed on the machine
CXX ?= g++
LD_LIBRARY_PATH := $(if $(LD_LIBRARY_PATH),$(SYSTEMC_HOME)/lib:$(SYSTEMC_HOME)/lib-linux64:$(LD_LIBRARY_PATH),$(SYSTEMC_HOME)/lib:$(SYSTEMC_HOME)/lib-linux64)
export LD_LIBRARY_PATH
LIBDIRS += -L$(SYSTEMC_HOME)/lib -L$(SYSTEMC_HOME)/lib-linux64

# chech the LD_LIBRARY_PATH
# $(info SYSTEMC_HOME: $(SYSTEMC_HOME))
# $(info LD_LIBRARY_PATH: $(LD_LIBRARY_PATH))
ifeq ($(LD_LIBRARY_PATH),)
$(error - Environment variable LD_LIBRARY_PATH must be defined)
endif


# Determine the director containing the source files from the path to this Makefile
PWD := $(shell pwd)
# SOURCE_DIR = $(subst syn,src/,$(PWD))
# TB_DIR = $(subst syn,tb/,$(PWD))
# INC_DIR = $(subst syn,inc/,$(PWD))
SOURCE_DIR = ../src/
TB_DIR = ../tb/
INC_DIR = ../inc/
# INCDMA_DIR = $(subst syn,inc/core/systems,$(PWD))

SRCS := $(wildcard $(SOURCE_DIR)*.cpp)
SRCS += $(wildcard $(TB_DIR)*.cpp)


SYS_FILE ?= ./dummy_sysc_catapult_dma64/dummy_sysc_catapult.v1/scverify/concat_sim_dummy_sysc_catapult_v_msim/system.cpp.cxxts

INCDIRS += -I$(SOURCE_DIR) -I$(TB_DIR) -I$(INC_DIR)

INCDIRS += -I$(SYSTEMC_HOME)/include -I$(SYSTEMC_HOME)/src 
INCDIRS += -I$(MATCHLIB_TOOL)/../include
# INCDIRS += -I$(MATCHLIB_TOOL)/matchlib_examples/include
INCDIRS += -I$(CONNECTIONS_HOME)/include
INCDIRS += -I$(MATCHLIB_HOME)/cmod/include
INCDIRS += -I$(BOOST_HOME)/include
INCDIRS += -I$(RAPIDJSON_HOME)/include
INCDIRS += -I$(AC_TYPES)/include
INCDIRS += -I$(AC_MATH)/include
INCDIRS += -I$(AC_SIMUTILS)/include

CPPFLAGS += $(INCDIRS)
CPPFLAGS += -DSC_INCLUDE_DYNAMIC_PROCESSES 
CPPFLAGS += -DSEGMENT_BURST_SIZE=16
CXXFLAGS += -D__CUSTOM_SIM__
CXXFLAGS += -D__MATCHLIB_CONNECTIONS__
CXXFLAGS += -DHLS_CATAPULT

LIBS += -lsystemc -lpthread



CATAPULT_PRODUCT = ultra

HW_ROOT ?= $(shell realpath ../..)
ESP_ROOT ?= $(error ESP_ROOT is not set. Please export it before running make.)

.PHONY: all build clean sim_clean sim hls
build: sim_sc_syn

all: hls sim

trace.vcd: sim_sc_syn
	-@echo "Starting execution in directory `pwd`"
	./$^

## BEHAVIORAL SIMULATION

# sim_sc_syn: $(SRCS) 
# 	$(CXX) $(CXXFLAGS) -DHLS_READY -DCONNECTIONS_ACCURATE_SIM $(CPPFLAGS) $(LIBDIRS) $(SRCS) -o $@ $(LIBS)

# sim_sc_syn_tlm: $(SRCS) 
# 	$(CXX) $(CXXFLAGS) -DHLS_READY -DCONNECTIONS_FAST_SIM $(CPPFLAGS) $(LIBDIRS) $(SRCS) -o $@ $(LIBS)

sim_sc: $(SRCS) 
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(LIBDIRS) $(SRCS) -o $@ $(LIBS) -DDMA_WIDTH=$(DMA_WIDTH)

bev-sim-clean:
	rm -f sim_sc
	
# This target is used to run behavioral simulation
# of the non-synthesizable version of the design for any input sentence
bev-sim: sim_sc
	-@echo "Starting execution in directory `pwd`"
	./$^ 


##############

## HLS

#This target is used to run hls

hls-clean:
	rm -rf dummy_sysc_catapult_dma64
	rm -f catapult.log
	rm -f dummy_sysc_catapult_dma64.ccs
	rm -f sim_sc

hls:
	@echo '   ' ;if test -e ./dummy_sysc_catapult_dma64; then \
		rm -r dummy_sysc_catapult_dma64; \
	fi;
	catapult -product $(CATAPULT_PRODUCT) -shell -f ./build_prj_top.tcl

##############

## RTL SIMULATION

#This target is used to run RTL-simulation
#for any input sentence
rtl_sim:
	date +%Y%m%d_%H%M%S > simulation_time.txt
	if test -e $(SYS_FILE); then \
		rm $(SYS_FILE) & \
	fi;
	@echo '   ' ;export IMM=$(@:-sim=); \
	if test -e ./dummy_sysc_catapult_dma64; then \
		SENTENCE=$(@:-sim=) catapult -product $(CATAPULT_PRODUCT) -shell -f ./rtl_sim.tcl; \
	else \
		echo "Run HLS before RTL-simulation"; \
	fi;
	date +%Y%m%d_%H%M%S >> simulation_time.txt

# search for the concatenated RTL and install it to ESP
install:
	@if test -e $(ESP_ROOT)/tech/virtex7/acc/dummy_sysc_catapult/dummy_sysc_catapult_dma64; then \
		echo "Found existing accelerator at $(ESP_ROOT)/tech/virtex7/acc/dummy_sysc_catapult/dummy_sysc_catapult_dma64"; \
		read -p "Do you want to remove it? (y/N): " confirm; \
		if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
			echo "Removing old accelerator..."; \
			rm -rf $(ESP_ROOT)/tech/virtex7/acc/dummy_sysc_catapult/dummy_sysc_catapult_dma64; \
		else \
			echo "Installation cancelled."; \
			exit 1; \
		fi \
	fi
	mkdir -p $(ESP_ROOT)/tech/virtex7/acc/dummy_sysc_catapult/dummy_sysc_catapult_dma64
	cp ../dummy_sysc_catapult.xml $(ESP_ROOT)/tech/virtex7/acc/dummy_sysc_catapult/.
	cp dummy_sysc_catapult_dma64/dummy_sysc_catapult.v1/concat_dummy_sysc_catapult.v $(ESP_ROOT)/tech/virtex7/acc/dummy_sysc_catapult/dummy_sysc_catapult_dma64/dummy_sysc_catapult_dma64.v
	sed -i 's/module dummy_sysc_catapult/module dummy_sysc_catapult_dma64/g' $(ESP_ROOT)/tech/virtex7/acc/dummy_sysc_catapult/dummy_sysc_catapult_dma64/dummy_sysc_catapult_dma64.v
	sed -i 's/Design Unit:    dummy_sysc_catapult/Design Unit:    dummy_sysc_catapult_dma64/g' $(ESP_ROOT)/tech/virtex7/acc/dummy_sysc_catapult/dummy_sysc_catapult_dma64/dummy_sysc_catapult_dma64.v


# SHELL = /bin/bash
# install:
# 	@mkdir -p $(RTL_OUT)
# ifneq ("$(TARGET_NAME)","")
# 	@NAME_SHORT=$(TARGET_NAME:_catapult=); \
# 	cp ../$$NAME_SHORT.xml $(RTL_OUT)/$(TARGET_NAME).xml; \
# 	for prj in $(TARGET_NAME)_*; do \
# 		if [[ $$prj == *".ccs"* ]]; then \
# 			continue; \
# 		fi; \
# 		PRJ_NAME=$$prj; \
# 		indir=$$prj/$(TARGET_NAME).v1; \
# 		infile=$$indir/concat_$(TARGET_NAME).v; \
# 		outdir=$(RTL_OUT)/$$PRJ_NAME; \
# 		outfile=$(RTL_OUT)/$$PRJ_NAME/$$PRJ_NAME.v; \
# 		mkdir -p $$outdir; \
# 		cp $$infile $$outfile; \
# 		sed -i "s/module $(TARGET_NAME)/module $$PRJ_NAME/g" $$outfile; \
# 		sed -i "s/Design Unit:    $(TARGET_NAME)/Design Unit:    $$PRJ_NAME/g" $$outfile; \
# 	done
# endif
.PHONY: install

# Non-interactive install: remove existing accelerator if present, then install.
# This is the target check_workspace.sh uses — no 'read -p' prompt.
install-force:
	@if test -e $(ESP_ROOT)/tech/virtex7/acc/dummy_sysc_catapult/dummy_sysc_catapult_dma64; then \
		echo "Removing existing accelerator at $(ESP_ROOT)/tech/virtex7/acc/dummy_sysc_catapult/dummy_sysc_catapult_dma64"; \
		rm -rf $(ESP_ROOT)/tech/virtex7/acc/dummy_sysc_catapult/dummy_sysc_catapult_dma64; \
	fi
	mkdir -p $(ESP_ROOT)/tech/virtex7/acc/dummy_sysc_catapult/dummy_sysc_catapult_dma64
	cp ../dummy_sysc_catapult.xml $(ESP_ROOT)/tech/virtex7/acc/dummy_sysc_catapult/.
	cp dummy_sysc_catapult_dma64/dummy_sysc_catapult.v1/concat_dummy_sysc_catapult.v $(ESP_ROOT)/tech/virtex7/acc/dummy_sysc_catapult/dummy_sysc_catapult_dma64/dummy_sysc_catapult_dma64.v
	sed -i 's/module dummy_sysc_catapult/module dummy_sysc_catapult_dma64/g' $(ESP_ROOT)/tech/virtex7/acc/dummy_sysc_catapult/dummy_sysc_catapult_dma64/dummy_sysc_catapult_dma64.v
	sed -i 's/Design Unit:    dummy_sysc_catapult/Design Unit:    dummy_sysc_catapult_dma64/g' $(ESP_ROOT)/tech/virtex7/acc/dummy_sysc_catapult/dummy_sysc_catapult_dma64/dummy_sysc_catapult_dma64.v
	@echo "Accelerator installed to $(ESP_ROOT)/tech/virtex7/acc/dummy_sysc_catapult/"
.PHONY: install-force

clean:
	rm -rf dummy_sysc_catapult_dma64
	rm -rf catapult_cache*
	rm -f catapult.pinfo
	rm -f log_dir_data.txt
	rm -f log_dir_names.txt
	rm -f trace.vcd
	rm -f transcript
	rm -f catapult.log
	rm -f vsim_stacktrace.vstf
	rm -f design_checker_constraints.tcl
	rm -f design_checker_pre_build.tcl
.PHONY: clean

distclean: clean
	@rm -rf $(RTL_OUT)
.PHONY: distclean
