// SPDX-FileCopyrightText: 2022 Old IAMR Repo Developers
//
// SPDX-License-Identifier: BSD-3-Clause

#include <cstdio>

#include <AMReX_CArena.H>
#include <AMReX_REAL.H>
#include <AMReX_Utility.H>
#include <AMReX_Amr.H>
#include <AMReX_ParmParse.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_BLProfiler.H>

#ifdef AMREX_USE_EB
#include <AMReX_EB2.H>
#include <AMReX_AmrLevel.H>
#endif

using namespace amrex;

#ifdef AMREX_USE_EB
//skipping header file and just declaring eb2 init fn here as in CNS for now
void initialize_EB2 (const Geometry& geom, int required_level, int max_level);
#endif

amrex::LevelBld* getLevelBld ();

int
main (int   argc,
      char* argv[])
{
    amrex::Initialize(argc,argv);

    BL_PROFILE_REGION_START("main()");
    BL_PROFILE_VAR("main()", pmain);

    const Real run_strt = ParallelDescriptor::second();

    int  max_step;
    int  num_steps;
    Real strt_time;
    Real stop_time;
    Real stop_interval;

    ParmParse pp;

    max_step  = -1;
    num_steps = -1;
    strt_time =  0.0;
    stop_time = -1.0;
    stop_interval = 0.;

    pp.query("max_step",  max_step);
    pp.query("num_steps", num_steps);
    pp.query("strt_time", strt_time);
    pp.query("stop_time", stop_time);
    pp.query("stop_interval", stop_interval);

    if (strt_time < 0.0)
    {
        amrex::Abort("MUST SPECIFY a non-negative strt_time");
    }

    if (max_step < 0 && stop_time < 0)
    {
        amrex::Abort("Exiting because neither max_step nor stop_time is non-negative.");
    }

    Amr* amrptr = new Amr(getLevelBld());
    //    Amr amr;
#ifdef AMREX_USE_EB
    // EB should calculate centroids as well as volume and area fractions
    AmrLevel::SetEBSupportLevel(EBSupport::full);

    // Set grow cells for basic, volume, full
    // Look in weighted SRSD paper for explanation for why this is 5
    AmrLevel::SetEBMaxGrowCells(5,5,5);

    int max_coarsening_level = 100;
    pp.query("max_coarsening_level", max_coarsening_level);
    initialize_EB2(amrptr->Geom(amrptr->maxLevel()), amrptr->maxLevel(),
           max_coarsening_level);
#endif

    amrptr->init(strt_time,stop_time);

    // This feature stop the simulation at a specific time
    // after the physical time of the checkpoint file
    if (stop_interval > 0.) stop_time = amrptr->cumTime() + stop_interval;

    if (num_steps > 0)
    {
        if (max_step < 0)
        {
            max_step = num_steps + amrptr->levelSteps(0);
        }
        else
        {
            max_step = std::min(max_step, num_steps + amrptr->levelSteps(0));
        }

    amrex::Print() << "Using effective max_step = " << max_step << '\n';
    }
    //
    // If we set the regrid_on_restart flag and if we are *not* going to take
    // a time step then we want to go ahead and regrid here.
    //
    if (Amr::RegridOnRestart())
    {
        if (    (amrptr->levelSteps(0) >= max_step ) ||
                ( (stop_time >= 0.0) &&
                  (amrptr->cumTime() >= stop_time)  )    )
        {
            //
            // Regrid only!
            //
            amrptr->RegridOnly(amrptr->cumTime());
        }
    }

    while ( amrptr->okToContinue()                            &&
           (amrptr->levelSteps(0) < max_step || max_step < 0) &&
           (amrptr->cumTime() < stop_time || stop_time < 0.0) )
    {
        amrptr->coarseTimeStep(stop_time);
    }
    //
    // Write final checkpoint and plotfile.
    //
    if (amrptr->stepOfLastCheckPoint() < amrptr->levelSteps(0))
    {
        amrptr->checkPoint();
    }

    if (amrptr->stepOfLastPlotFile() < amrptr->levelSteps(0))
    {
        amrptr->writePlotFile();
    }

    delete amrptr;

    const int IOProc   = ParallelDescriptor::IOProcessorNumber();
    Real      run_stop = ParallelDescriptor::second() - run_strt;

    ParallelDescriptor::ReduceRealMax(run_stop,IOProc);

    amrex::Print() << "Run time = " << run_stop << std::endl;

    BL_PROFILE_VAR_STOP(pmain);
    BL_PROFILE_REGION_STOP("main()");
    BL_PROFILE_SET_RUN_TIME(run_stop);
    BL_PROFILE_FINALIZE();


    amrex::Finalize();

    return 0;
}
