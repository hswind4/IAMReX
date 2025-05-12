# SPDX-FileCopyrightText: 2022 Old IAMR Repo Developers; 2022 - 2025 Yadong Zeng<ruohai372@gmail.com>
#
# SPDX-License-Identifier: BSD-3-Clause

#AMREX_HOME defines the directory in which we will find the AMReX directory
AMREX_HOME       ?= ../../../amrex
AMREX_HYDRO_HOME ?= ../../../AMReX-Hydro

#TOP defines the directory in which we will find Source, Exec, etc.
TOP = ../..

#
# Variables for the user to set ...
#

PRECISION   = DOUBLE

DIM         = 2 # 3
COMP        = gnu

DEBUG       = FALSE
USE_MPI     = TRUE
USE_OMP     = FALSE
PROFILE     = FALSE

USE_CUDA = FALSE
#CUDA_ARCH = 80
#CUDA_LAUNCH_BLOCK_BLOCKING=1

USE_SENSEI_INSITU = FALSE

EBASE = amr

Blocs   := .

include $(TOP)/Exec/Make.IAMR
