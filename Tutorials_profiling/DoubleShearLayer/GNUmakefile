# SPDX-FileCopyrightText: 1998 - 2023 Berkeley Lab
# SPDX-FileCopyrightText: 2023 - 2025 Yadong Zeng<zdsjtu@gmail.com> & Dewen Liu<1803368907@qq.com>
#
# SPDX-License-Identifier: LicenseRef-OpenSource
# Modified from IAMR, originally developed at Lawrence Berkeley National Lab.
# Original source: https://github.com/AMReX-Fluids/IAMR

#AMREX_HOME defines the directory in which we will find the BoxLib directory
AMREX_HOME ?= ../../../amrex
AMREX_HYDRO_HOME ?= ../../../AMReX-Hydro

#TOP defines the directory in which we will find Source, Exec, etc.
TOP = ../..

#
# Variables for the user to set ...
#

PRECISION   = DOUBLE

DIM         = 2 # DIM
COMP        = gnu

DEBUG       = FALSE
USE_MPI     = FALSE # MPI
USE_OMP     = FALSE # OMP
PROFILE     = TRUE

USE_CUDA = TRUE # CUDA

USE_EB = TRUE

USE_VELOCITY = FALSE

EBASE = amr

Blocs   := .

include $(TOP)/Exec/Make.IAMR
