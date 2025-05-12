# SPDX-FileCopyrightText: 2022 Old IAMR Repo Developers
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

DIM         = 2
COMP        = gnu

DEBUG       = FALSE
USE_MPI     = FALSE
USE_OMP     = TRUE
PROFILE     = FALSE

USE_EB = TRUE

USE_VELOCITY = FALSE

EBASE = amr

Blocs   := .

include $(TOP)/Exec/Make.IAMR
