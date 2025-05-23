/*
 * SPDX-FileCopyrightText: 2022 Old IAMR Repo Developers; 2022 - 2025 Yadong Zeng<zdsjtu@gmail.com>
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef IAMR_Projection_H_
#define IAMR_Projection_H_

#include <AMReX_BC_TYPES.H>
#include <AMReX_BCRec.H>
#include <AMReX_Amr.H>
#include <AMReX_AmrLevel.H>
#include <RegType.H>
#include <SyncRegister.H>
#include <AMReX_FluxRegister.H>
#include <AMReX_MultiFabUtil.H>

#ifdef AMREX_USE_EB
#include <AMReX_EBFabFactory.H>
#endif

class Projection
{
protected:
  static const int maxlev = 10;
  static const int  LEVEL_PROJ    = 1001;
  static const int  INITIAL_VEL   = 1002;
  static const int  INITIAL_PRESS = 1003;
  static const int  INITIAL_SYNC  = 1004;
  static const int  SYNC_PROJ     = 1005;
  static const int  FILTER_P      = 1006;

public:

    Projection (amrex::Amr*   _parent,
                amrex::BCRec* _phys_bc,
                int    _do_sync_proj,
                int    /*_finest_level*/,
                int    _radius_grow );

    //
    // Set a level in the projector.
    //
    void install_level (int                   level,
                        amrex::AmrLevel*             level_data,
                        amrex::Vector< amrex::Vector<amrex::Real> >* _radius);
#ifdef AMREX_USE_EB
    void install_level (int                   level,
                        amrex::AmrLevel*             level_data,
                        amrex::Vector< amrex::Vector<amrex::Real> >* _radius,
            const amrex::EBFArrayBoxFactory& _ebfactory);

#endif
    //
    // Compute the level projection in NavierStokes::advance.
    //
    void level_project (int             level,
                        amrex::Real            time,
                        amrex::Real            dt,
                        amrex::Real            cur_pres_time,
                        const amrex::Geometry& geom,
                        amrex::MultiFab&       U_old,
                        amrex::MultiFab&       U_new,
                        amrex::MultiFab&       P_old,
                        amrex::MultiFab&       P_new,
                        amrex::MultiFab&       rho_half,
                        SyncRegister*   crse_sync_reg,
                        SyncRegister*   fine_sync_reg,
                        int             crse_dt_ratio,
                        int             iteration,
                        int             have_divu);

    // solve DG(correction to P_new) = -D G^perp p^(n-half)
    //  or   DG(correction to P_new) = -D G^perp p^(n-half) - D(U^n /dt)
    void filterP(int level,
                 const amrex::Geometry& geom,
                 amrex::MultiFab &P_old,
                 amrex::MultiFab &P_new,
                 amrex::MultiFab &U_old,
                 amrex::MultiFab &rho_half,
                 int     ** bc,
                 amrex::Real            time,
                 amrex::Real            dt,
                 int    have_divu);
    //
    // A sync project involving the coarse and fine level of a 2 level system.
    //
    void MLsyncProject (int                    c_lev,
                        amrex::MultiFab&       pres_crse,
                        amrex::MultiFab&       vel_crse,
                        amrex::MultiFab&       cc_rhs_crse,
                        amrex::MultiFab&       pres_fine,
                        amrex::MultiFab&       vel_fine,
                        amrex::MultiFab&       cc_rhs_fine,
                        amrex::MultiFab&       rho_crse,
                        amrex::MultiFab&       rho_fine,
                        amrex::MultiFab&       Vsync,
                        amrex::MultiFab&       V_corr,
                        amrex::MultiFab&       phi,
                        SyncRegister*          rhs_sync_reg,
                        SyncRegister*          crsr_sync_reg,
                        amrex::Real            dt_crse,
                        amrex::IntVect&        ratio,
                        int                    crse_iteration,
                        int                    crse_dt_ratio,
                        const amrex::Geometry& crse_geom);
    //
    // This projects an initial vorticity field to define a velocity field.
    //
    void initialVorticityProject (int c_lev);
    //
    // This computes the stream function, given the velocity field
    //  *AND ASSUMING* no-flow boundaries on all physical boundaries.
    //
    void getStreamFunction (amrex::Vector<std::unique_ptr<amrex::MultiFab> >&);
    //
    // The initial velocity projection in post_init.
    // This function ensures that the velocities are nondivergent.
    //
    void initialVelocityProject (int c_lev,
                                 amrex::Real divu_time,
                                 int  have_divu,
                                 int init_vel_iter = 1 );

    //
    // This function creates an initially hydrostatic pressure field
    //   in the case of nonzero gravity.
    //
    void initialPressureProject (int  c_lev);
    //
    // The velocity projection in post_init, which computes the initial
    // pressure used in the timestepping.
    //
    void initialSyncProject (int c_lev,
                             const amrex::Vector<amrex::MultiFab*>& sig,
                             amrex::Real      dt,
                             amrex::Real      strt_time,
                             int       have_divu);
    //
    // Multiply a MultiFab by radius for r-z coordinates.
    // These should really be protected.
    //
    void radMultScal (int level, amrex::MultiFab& mf) const;

    //
    // Read parameters from the input file.
    //
    static void Initialize ();

    static void Finalize ();

    //
    // Convert U to an Acceleration like quantity
    // Unew = (Unew - Uold)/alpha
    //
    static void ConvertUnew (amrex::MultiFab&       Unew,
                             amrex::MultiFab&       Uold,
                             amrex::Real            alpha,
                             const amrex::BoxArray& grids );
    //
    // Convert U to an Acceleration like quantity.
    // Unew = (Unew - Uold)/alpha
    //
    static void ConvertUnew (amrex::FArrayBox& Unew,
                             amrex::FArrayBox& Uold,
                             amrex::Real       alpha,
                             const amrex::Box& grd);
    //
    // This function scales variables at the start of a projection.
    //
    void scaleVar (amrex::MultiFab* sig,
                   int              sig_nghosts,
                   amrex::MultiFab* vel,
                   int              level) const;
    //
    // This function rescales variables at the end of a projection.
    //
    void rescaleVar (amrex::MultiFab* sig,
                     int              sig_nghosts,
                     amrex::MultiFab* vel,
                     int              level) const;

    void set_outflow_bcs (int        which_call,
                          const amrex::Vector<amrex::MultiFab*>& phi,
                          const amrex::Vector<amrex::MultiFab*>& Vel_in,
                          const amrex::Vector<amrex::MultiFab*>& Divu_in,
                          const amrex::Vector<amrex::MultiFab*>& Sig_in,
                          int        c_lev,
                          int        f_lev,
                          int        have_divu);

    void set_outflow_bcs_at_level (int          which_call,
                                   int          lev,
                                   int          c_lev,
                                   amrex::Box*         state_strip,
                                   amrex::Orientation* outFacesAtThisLevel,
                                   int          numOutFlowFaces,
                                   const amrex::Vector<amrex::MultiFab*>& phi,
                                   amrex::MultiFab*    Vel_in,
                                   amrex::MultiFab*    Divu_in,
                                   amrex::MultiFab*    Sig_in,
                                   int          have_divu,
                                   amrex::Real         gravity);

    void computeRhoG (amrex::FArrayBox*         rhoFab,
                      amrex::FArrayBox*         phiFab,
                      const amrex::Geometry&    geom,
                      amrex::Orientation*       outFaces,
                      int                       numOutFlowFaces,
                      amrex::Real               gravity);

    void putDown (const amrex::Vector<amrex::MultiFab*>& phi,
                  amrex::FArrayBox* phi_fine_strip,
                  int c_lev, int f_lev, const amrex::Orientation* outFaces,
                  int numOutFlowFaces, int ncStripWidth) const;
    //
    // Pointers to amrlevel and amr.
    //
    amrex::Amr*             parent;
    amrex::Vector<amrex::AmrLevel*> LevelData;
#ifdef AMREX_USE_EB
    // pointer to eb factory on each level
    // could potentially do away with this and use LevelData to get ebFactory...
    amrex::Vector<const amrex::EBFArrayBoxFactory*> ebfactory;
#endif
    //
    // The array of Radii, and number of radii ghost cells.
    //
    int                    radius_grow;
    amrex::Vector< amrex::Vector< amrex::Vector<amrex::Real> > * > radius;

    //
    // Boundary objects.
    //
    amrex::BCRec*                         phys_bc;

    int do_sync_proj;

    static int  verbose;
    static amrex::Real proj_tol;
    static amrex::Real sync_tol;
    static amrex::Real proj_abs_tol;
    static int  proj_2;
    static int  rho_wgt_vel_proj;
    static int  do_outflow_bcs;
    static int  add_vort_proj;

    void doMLMGNodalProjection (int c_lev, int nlevel,
                                const amrex::Vector<amrex::MultiFab*>& vel,
                                const amrex::Vector<amrex::MultiFab*>& phi,
                                const amrex::Vector<amrex::MultiFab*>& sig,
                                const amrex::Vector<amrex::MultiFab*>& rhcc,
                                const amrex::Vector<amrex::MultiFab*>& rhnd,
                                amrex::Real rel_tol, amrex::Real abs_tol,
                                bool increment_gp,
                                amrex::MultiFab* sync_resid_crse=nullptr,
                                amrex::MultiFab* sync_resid_fine=nullptr,
                                bool doing_initial_vortproj=false);

    // set velocity in ghost cells to zero except for inflow
    void set_boundary_velocity (int c_lev, int nlevel,
                                const amrex::Vector<amrex::MultiFab*>& vel,
                                bool inflowCorner) const;

};
#endif
