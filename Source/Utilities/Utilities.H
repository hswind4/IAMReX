/*
 * SPDX-FileCopyrightText: 2022 Old IAMR Repo Developers
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef IAMR_UTILITIES_H_
#define IAMR_UTILITIES_H_

#include <AMReX_FArrayBox.H>


AMREX_FORCE_INLINE
std::string
read_file(std::ifstream& in)
{
  return static_cast<std::stringstream const&>(
           std::stringstream() << in.rdbuf())
    .str();
}

void read_binary(
  const std::string& iname,
  size_t nx,
  size_t ny,
  size_t nz,
  size_t ncol,
  amrex::Vector<amrex::Real>& data);

void read_csv(
  const std::string& iname,
  size_t nx,
  size_t ny,
  size_t nz,
  amrex::Vector<amrex::Real>& data);

// -----------------------------------------------------------
// Search for the closest index in an array to a given value
// using the bisection technique.
// INPUTS/OUTPUTS:
// xtable(0:n-1) => array to search in (ascending order)
// n             => array size
// x             => x location
// idxlo        <=> output st. xtable(idxlo) <= x < xtable(idxlo+1)
// -----------------------------------------------------------
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
locate(const amrex::Real* xtable, const int n, const amrex::Real& x, int& idxlo)
{
  // If x is out of bounds, return boundary index
  if (x >= xtable[n - 1]) {
    idxlo = n - 1;
    return;
  }
  if (x <= xtable[0]) {
    idxlo = 0;
    return;
  }

  // Do the bisection
  idxlo = 0;
  int idxhi = n - 1;
  bool notdone = true;
  while (notdone) {
    if (idxhi - idxlo <= 1) {
      notdone = false;
    } else {
      const int idxmid = (idxhi + idxlo) / 2;
      if (x >= xtable[idxmid]) {
        idxlo = idxmid;
      } else {
        idxhi = idxmid;
      }
    }
  }
}


#endif
