# SPDX-FileCopyrightText: 2022 - 2025 Yadong Zeng<zdsjtu@gmail.com> & Dewen Liu<1803368907@qq.com>
#
# SPDX-License-Identifier: BSD-3-Clause

#AMREX_HOME defines the directory in which we will find the BoxLib directory
AMREX_HOME ?= ../../../amrex
AMREX_HYDRO_HOME ?= ../../../AMReX-Hydro

#TOP defines the directory in which we will find Source, Exec, etc.
TOP = ../..

#
# Variables for the user to set ...
#

PRECISION = DOUBLE

DEBUG	  = FALSE

USE_MPI     = TRUE
USE_OMP     = TRUE
PROFILE     = TRUE

DIM       = 2 # DIM

COMP	  = g++

USE_PARTICLES   = TRUE

EBASE     = amr

Blocs   := .

include $(TOP)/Exec/Make.IAMR
