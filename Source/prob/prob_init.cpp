// SPDX-FileCopyrightText: 2022 Old IAMR Repo Developers; 2022 - 2025 Yadong Zeng<zdsjtu@gmail.com>
//
// SPDX-License-Identifier: BSD-3-Clause

#include <NavierStokes.H>
#include <AMReX_ParmParse.H>

using namespace amrex;

int NavierStokes::probtype = -1;
amrex::Real NavierStokes::ub = 2.0;
amrex::Real NavierStokes::shearrate = 3.0;

// For now, define pi here, but maybe later make iamr_constants.H
namespace {
  constexpr Real Pi    = 3.141592653589793238462643383279502884197;
  constexpr Real TwoPi = 2.0 * 3.141592653589793238462643383279502884197;
}

//
// Initialize state and pressure with problem-specific data
//
void NavierStokes::prob_initData ()
{

    //
    // Create struct to hold initial conditions parameters
    //
    InitialConditions IC;

    //
    // Read problem parameters from inputs file
    //
    ParmParse pp("prob");
    ParmParse pp2("ns");

    pp.query("probtype",probtype);
    pp.query("ub",ub);
    pp.query("shearrate",shearrate);
    pp2.query("fluid_rho",IC.density);
    pp.query("direction",IC.direction);
    pp.query("interface_width",IC.interface_width);

    Vector<Real> velocity(AMREX_SPACEDIM, 0.);
    pp.queryarr("velocity_ic",velocity,0,AMREX_SPACEDIM);
    AMREX_D_TERM(IC.v_x = velocity[0];,
         IC.v_y = velocity[1];,
         IC.v_z = velocity[2];);

    pp.query("blob_radius",IC.blob_radius);
    Vector<Real> blob_center(AMREX_SPACEDIM, 0.);
    pp.queryarr("blob_center",blob_center,0,AMREX_SPACEDIM);
    AMREX_D_TERM(IC.blob_x = blob_center[0];,
         IC.blob_y = blob_center[1];,
         IC.blob_z = blob_center[2];);

    // For Rayleigh-Taylor problem
    pp.query("rho_1",IC.rho_1);
    pp.query("rho_2",IC.rho_2);
    pp.query("tra_1",IC.tra_1);
    pp.query("tra_2",IC.tra_2);
    pp.query("perturbation_amplitude",IC.pertamp);

    // for Taylor-Green
    pp.query("velocity_factor",IC.v_x);
    pp.query("a", IC.a);
    pp.query("b", IC.b);
    pp.query("c", IC.c);

    // for Convected Vortex
    if (probtype == 8 )
    {
        IC.a = 0.5; // x-position of vortex
        IC.b = 0.5; // y-position of vortex
        IC.c = 0.07;// radius of vortex

        pp.query("xvort", IC.a);
        pp.query("yvort", IC.b);
        pp.query("rvort", IC.c);
        pp.query("forcevort", IC.forcevort);
        pp.query("meanFlowDir", IC.meanFlowDir);
        pp.query("meanFlowMag", IC.meanFlowMag);
    }

    //
    // ls related
    //
    IC.do_phi = do_phi;
    if (do_phi) {
      IC.Density = Density; // Bug to be fixed: IC.Density is density value, yet Density is the scomp.
      IC.phicomp = phicomp;
      IC.rho_w = rho_w;
      IC.rho_a = rho_a;
    }

    //
    // Fill state and, optionally, pressure
    //
    MultiFab& P_new = get_new_data(Press_Type);
    MultiFab& S_new = get_new_data(State_Type);
    const int nscal = NUM_STATE-Density;

    S_new.setVal(0.0);
    P_new.setVal(0.0);

    // Integer indices of the lower left and upper right corners of the
    // valid region of the entire domain.
    Box const&  domain = geom.Domain();
    auto const&     dx = geom.CellSizeArray();
    // Physical coordinates of the lower left corner of the domain
    auto const& problo = geom.ProbLoArray();
    // Physical coordinates of the upper right corner of the domain
    auto const& probhi = geom.ProbHiArray();

#ifdef _OPENMP
#pragma omp parallel  if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(S_new,TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& vbx = mfi.tilebox();

        if ( 1 == probtype )
        {
            //
            // Start from rest, constant density of 1
            // Introduced for LidDrivenCavity problem
            //
            S_new[mfi].setVal<RunOn::Gpu>(1.0,Density);
        }
        else if ( 2 == probtype || 6 == probtype )
        {
            init_bubble(vbx, P_new.array(mfi), S_new.array(mfi, Xvel),
                        S_new.array(mfi, Density), nscal,
                        domain, dx, problo, probhi, IC);
        }
        else if ( 3 == probtype )
        {
            init_jump(vbx, P_new.array(mfi), S_new.array(mfi, Xvel),
                      S_new.array(mfi, Density), nscal,
                      domain, dx, problo, probhi, IC);
        }
        else if ( 4 == probtype )
        {
            init_constant_vel_rho(vbx, P_new.array(mfi), S_new.array(mfi, Xvel),
                                  S_new.array(mfi, Density), nscal,
                                  domain, dx, problo, probhi, IC);
        }

        else if ( 97 == probtype )
        {
            init_channel(vbx, P_new.array(mfi), S_new.array(mfi, Xvel),
                                  S_new.array(mfi, Density), nscal,
                                  domain, dx, problo, probhi, IC);
        }

        else if ( 5 == probtype )
        {
            init_DoubleShearLayer(vbx, P_new.array(mfi), S_new.array(mfi, Xvel),
                                  S_new.array(mfi, Density), nscal,
                                  domain, dx, problo, probhi, IC);
        }
        else if ( 7 == probtype )
        {
            init_Euler(vbx, P_new.array(mfi), S_new.array(mfi, Xvel),
                       S_new.array(mfi, Density), nscal,
                       domain, dx, problo, probhi, IC);
        }
        else if ( 8 == probtype )
        {
            init_ConvectedVortex(vbx, P_new.array(mfi), S_new.array(mfi, Xvel),
                                 S_new.array(mfi, Density), nscal,
                                 domain, dx, problo, probhi, IC);
        }
        else if ( 10 == probtype )
        {
            init_RayleighTaylor(vbx, P_new.array(mfi), S_new.array(mfi, Xvel),
                                S_new.array(mfi, Density), nscal,
                                domain, dx, problo, probhi, IC);
        }
        else if ( 11 == probtype )
        {
            init_TaylorGreen(vbx, P_new.array(mfi), S_new.array(mfi, Xvel),
                             S_new.array(mfi, Density), nscal,
                             domain, dx, problo, probhi, IC);
        }
        else if ( 98 == probtype ) // for PVF case
        {
            init_particles(vbx, P_new.array(mfi), S_new.array(mfi, Xvel),
                             S_new.array(mfi, Density), nscal,
                             domain, dx, problo, probhi, IC);
        }
        else if ( 99 == probtype ) // ls related
        {
            init_rsv(vbx, P_new.array(mfi), S_new.array(mfi, Xvel),
                        S_new.array(mfi, Density), nscal,
                        domain, dx, problo, probhi, IC);
        }
        else if ( 100 == probtype )
        {
            init_RayleighTaylor_LS(vbx, P_new.array(mfi), S_new.array(mfi, Xvel),
                                S_new.array(mfi, Density), nscal,
                                domain, dx, problo, probhi, IC);
        }
        else if ( 101 == probtype )
        {
            init_BreakingWave(vbx, P_new.array(mfi), S_new.array(mfi, Xvel),
                                S_new.array(mfi, Density), nscal,
                                domain, dx, problo, probhi, IC);
        }
        else if ( 102 == probtype ||  103 == probtype ) // Sphere near the channel wall
        {
           SphereNearWall(vbx, P_new.array(mfi), S_new.array(mfi, Xvel),
                             S_new.array(mfi, Density), nscal,
                            domain, dx, problo, probhi, IC);
        }
        else if ( 104 == probtype ) // Falling Sphere
        {
           FallingSphere(vbx, P_new.array(mfi), S_new.array(mfi, Xvel),
                             S_new.array(mfi, Density), nscal,
                            domain, dx, problo, probhi, IC);
        }
        else
        {
            amrex::Abort("NavierStokes::prob_init: unknown probtype");
        }
    }
}

void NavierStokes::init_bubble (Box const& vbx,
                Array4<Real> const& /*press*/,
                Array4<Real> const& vel,
                Array4<Real> const& scal,
                const int nscal,
                Box const& domain,
                GpuArray<Real, AMREX_SPACEDIM> const& dx,
                GpuArray<Real, AMREX_SPACEDIM> const& problo,
                GpuArray<Real, AMREX_SPACEDIM> const& /*probhi*/,
                InitialConditions IC)
{
  const auto domlo = amrex::lbound(domain);

  // Make local copy capturable by device lambda
  bool have_temp = probtype == 6;

  amrex::ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {
    Real x = problo[0] + (i - domlo.x + 0.5)*dx[0];
    Real y = problo[1] + (j - domlo.y + 0.5)*dx[1];

    //
    // Fill Velocity
    //
    vel(i,j,k,0) = IC.v_x;
    vel(i,j,k,1) = IC.v_y;

#if (AMREX_SPACEDIM == 3)
    Real z = problo[2] + (k - domlo.z + 0.5)*dx[2];

    vel(i,j,k,2) = IC.v_z;
#endif

    Real dist = std::sqrt( (x-IC.blob_x)*(x-IC.blob_x)
              + (y-IC.blob_y)*(y-IC.blob_y)
#if (AMREX_SPACEDIM == 3)
              + (z-IC.blob_z)*(z-IC.blob_z)
#endif
              );
    //
    // Scalars, ordered as Density, Tracer(s), Temp (if using)
    //

    // Tracers
    scal(i,j,k,1) = dist < IC.blob_radius ? 1.0 : 0.0;
    for ( int nt=2; nt<nscal; nt++)
    {
      scal(i,j,k,nt) = dist < IC.blob_radius ? 1.0 : 0.0;
    }

    if ( have_temp )
    {
      // Density for Hot/less dense bubble rising, assuming IC.density > 1
      scal(i,j,k,0) = 1.0/IC.density + 0.5*(1.0 - 1.0/IC.density)*(1.0 + std::tanh(40.*(dist - IC.blob_radius)/IC.interface_width));
      //Temp
      scal(i,j,k,nscal-1) = 1/scal(i,j,k,0);
    }
    else
    {
      // Density for dense bubble falling, assuming IC.density > 1
      scal(i,j,k,0) = 1.0 + 0.5*(IC.density-1.0)*(1.0-std::tanh(30.*(dist-IC.blob_radius)/IC.interface_width));
      // No Temperature field
    }

  });
}

//
// ls related
// 
void NavierStokes::init_rsv (Box const& vbx,
                Array4<Real> const& /*press*/,
                Array4<Real> const& vel,
                Array4<Real> const& scal,
                const int nscal,
                Box const& domain,
                GpuArray<Real, AMREX_SPACEDIM> const& dx,
                GpuArray<Real, AMREX_SPACEDIM> const& problo,
                GpuArray<Real, AMREX_SPACEDIM> const& /*probhi*/,
                InitialConditions IC)
{

  BL_ASSERT(IC.do_phi==1);

  const auto domlo = amrex::lbound(domain);

  amrex::ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {
    Real x = problo[0] + (i - domlo.x + 0.5)*dx[0];
    Real y = problo[1] + (j - domlo.y + 0.5)*dx[1];

    //
    // Fill Velocity
    //
    vel(i,j,k,0) = - std::sin(Pi*x)*std::sin(Pi*x)*std::sin(2*Pi*y);
    vel(i,j,k,1) =   std::sin(2*Pi*x)*std::sin(Pi*y)*std::sin(Pi*y);

#if (AMREX_SPACEDIM == 3)
    vel(i,j,k,2) = 0.0;
#endif

    //
    // Scalars, ordered as Density, Tracer(s), Temp (if using), ls
    //

    // All Tracers are set here
    for ( int nt=0; nt<nscal; nt++)
    {
      scal(i,j,k,nt) = 1.0;
    }
    
    // Initialize the LS function if do_phi
    if (IC.do_phi) {
      scal(i,j,k,IC.phicomp-IC.Density) = IC.blob_radius - std::sqrt( (x-IC.blob_x)*(x-IC.blob_x)
              + (y-IC.blob_y)*(y-IC.blob_y));
    } 

  });
}

//
// ls related
// 
void NavierStokes::set_rsv_vel (Box const& vbx,
                Array4<Real> const& vel,
                Box const& domain,
                GpuArray<Real, AMREX_SPACEDIM> const& dx,
                GpuArray<Real, AMREX_SPACEDIM> const& problo,
                GpuArray<Real, AMREX_SPACEDIM> const& /*probhi*/,
                InitialConditions IC,
                Real time)
{
  const auto domlo = amrex::lbound(domain);
  amrex::ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {
    Real x = problo[0] + (i - domlo.x + 0.5)*dx[0];
    Real y = problo[1] + (j - domlo.y + 0.5)*dx[1];

    //
    // Fill Velocity
    //
    vel(i,j,k,0) = - std::sin(Pi*x)*std::sin(Pi*x)*std::sin(2*Pi*y)*std::cos(Pi*time/IC.totalTimeRsv);
    vel(i,j,k,1) =   std::sin(2*Pi*x)*std::sin(Pi*y)*std::sin(Pi*y)*std::cos(Pi*time/IC.totalTimeRsv);

#if (AMREX_SPACEDIM == 3)
    vel(i,j,k,2) = 0.0;
#endif

  });
}

//
// diffused ib
// 
void NavierStokes::set_initial_phi_nodal (Box const& bx,
                Array4<Real> const& phi_nodal,
                Box const& domain,
                GpuArray<Real, AMREX_SPACEDIM> const& dx,
                GpuArray<Real, AMREX_SPACEDIM> const& problo,
                GpuArray<Real, AMREX_SPACEDIM> const& /*probhi*/,
                InitialConditions IC,
                Real time)
{
  
  BL_ASSERT(AMREX_SPACEDIM == 3);
  BL_ASSERT(probtype == 98);

  const auto domlo = amrex::lbound(domain);
  amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {
    Real x = problo[0] + (i - domlo.x)*dx[0];
    Real y = problo[1] + (j - domlo.y)*dx[1];
    Real z = problo[2] + (k - domlo.z)*dx[2];

    phi_nodal(i,j,k) = std::sqrt( (x-IC.blob_x)*(x-IC.blob_x)
              + (y-IC.blob_y)*(y-IC.blob_y)  + (z-IC.blob_z)*(z-IC.blob_z)) - IC.blob_radius;;
    phi_nodal(i,j,k) = phi_nodal(i,j,k) / IC.blob_radius;

  });

}

// Sphere near the channel wall
void  NavierStokes::SphereNearWall (amrex::Box const& vbx,
               amrex::Array4<amrex::Real> const& press,
               amrex::Array4<amrex::Real> const& vel,
               amrex::Array4<amrex::Real> const& scal,
               int nscal,
               amrex::Box const& domain,
               amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
               amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
               amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& probhi,
               InitialConditions IC)
{
  BL_ASSERT(AMREX_SPACEDIM == 3);
  const auto domlo = amrex::lbound(domain);
  // Initial velocity of flow field
  amrex::ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {

    //
    // Scalars, ordered as Density, Tracer(s), Temp (if using)
    //

    vel(i,j,k,0) = 0.0;
    vel(i,j,k,1) = 0.0;
    vel(i,j,k,2) = 0.0;

    scal(i,j,k,0) = IC.density;

    // Tracers
    scal(i,j,k,1) = 0.0;


  });
}

// Falling Sphere
void  NavierStokes::FallingSphere (amrex::Box const& vbx,
               amrex::Array4<amrex::Real> const& press,
               amrex::Array4<amrex::Real> const& vel,
               amrex::Array4<amrex::Real> const& scal,
               int nscal,
               amrex::Box const& domain,
               amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
               amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
               amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& probhi,
               InitialConditions IC)
{
  BL_ASSERT(AMREX_SPACEDIM == 3);
  const auto domlo = amrex::lbound(domain);
  // Initial velocity of flow field
  amrex::ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {

    //
    // Scalars, ordered as Density, Tracer(s), Temp (if using)
    //

    vel(i,j,k,0) = 0.0;
    vel(i,j,k,1) = 0.0;
    vel(i,j,k,2) = 0.0;

    scal(i,j,k,0) = IC.density;

    // Tracers
    scal(i,j,k,1) = 0.0;


  });
}

void NavierStokes::init_constant_vel_rho (Box const& vbx,
                      Array4<Real> const& /*press*/,
                      Array4<Real> const& vel,
                      Array4<Real> const& scal,
                      const int nscal,
                      Box const& domain,
                      GpuArray<Real, AMREX_SPACEDIM> const& dx,
                      GpuArray<Real, AMREX_SPACEDIM> const& problo,
                      GpuArray<Real, AMREX_SPACEDIM> const& /*probhi*/,
                      InitialConditions IC)
{
  const auto domlo = amrex::lbound(domain);

  amrex::ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {
    Real x = problo[0] + (i - domlo.x + 0.5)*dx[0];
    Real y = problo[1] + (j - domlo.y + 0.5)*dx[1];

    //
    // Fill Velocity
    //
    vel(i,j,k,0) = IC.v_x;
    vel(i,j,k,1) = IC.v_y;

#if (AMREX_SPACEDIM == 3)
    Real z = problo[2] + (k - domlo.z + 0.5)*dx[2];

    vel(i,j,k,2) = IC.v_z;
#endif

    Real dist = std::sqrt( (x-IC.blob_x)*(x-IC.blob_x)
              + (y-IC.blob_y)*(y-IC.blob_y)
#if (AMREX_SPACEDIM == 3)
              + (z-IC.blob_z)*(z-IC.blob_z)
#endif
              );
    //
    // Scalars, ordered as Density, Tracer(s), Temp (if using)
    //
    scal(i,j,k,0) = IC.density;

    // Tracers
    scal(i,j,k,1) = 0.5*(1.0-std::tanh(25.*(dist-IC.blob_radius)/IC.interface_width));
    for ( int nt=2; nt<nscal; nt++)
    {
      // scal(i,j,k,nt) = dist < IC.blob_radius ? 1.0 : 0.0;
      scal(i,j,k,nt) = 0.0;
    }
  });
}

void NavierStokes::init_channel (Box const& vbx,
                      Array4<Real> const& /*press*/,
                      Array4<Real> const& vel,
                      Array4<Real> const& scal,
                      const int nscal,
                      Box const& domain,
                      GpuArray<Real, AMREX_SPACEDIM> const& dx,
                      GpuArray<Real, AMREX_SPACEDIM> const& problo,
                      GpuArray<Real, AMREX_SPACEDIM> const& probhi,
                      InitialConditions IC)
{
  
  BL_ASSERT(AMREX_SPACEDIM == 3);
  const auto domlo = amrex::lbound(domain);

  amrex::ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {
    // Real x = problo[0] + (i - domlo.x + 0.5)*dx[0];
    Real y = problo[1] + (j - domlo.y + 0.5)*dx[1];
    // Real z = problo[2] + (k - domlo.z + 0.5)*dx[2];

    //
    // Fill Velocity
    //
    vel(i,j,k,0) = 0.0;
    vel(i,j,k,1) = 0.0;

    Real w_b = 18.5;
    const Real Ly    = (probhi[1] - problo[1]);
    vel(i,j,k,2) = w_b * (1.0 - std::pow((y/Ly - 1.0),2.0));

    //
    // Scalars, ordered as Density, Tracer(s), Temp (if using)
    //
    scal(i,j,k,0) = 1.0;

    // Tracers
    scal(i,j,k,1) = 0.0;

  });
}

void NavierStokes::init_jump (Box const& vbx,
                      Array4<Real> const& /*press*/,
                      Array4<Real> const& vel,
                      Array4<Real> const& scal,
                      const int nscal,
                      Box const& domain,
                      GpuArray<Real, AMREX_SPACEDIM> const& dx,
                      GpuArray<Real, AMREX_SPACEDIM> const& problo,
                      GpuArray<Real, AMREX_SPACEDIM> const& /*probhi*/,
                      InitialConditions IC)
{
  const auto domlo = amrex::lbound(domain);

  amrex::ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {
    Real x = problo[0] + (i - domlo.x + 0.5)*dx[0];
    Real y = problo[1] + (j - domlo.y + 0.5)*dx[1];

    //
    // Fill Velocity
    //
    vel(i,j,k,0) = IC.v_x;
    vel(i,j,k,1) = IC.v_y;

#if (AMREX_SPACEDIM == 3)
    Real z = problo[2] + (k - domlo.z + 0.5)*dx[2];

    vel(i,j,k,2) = IC.v_z;
#endif

    Real dist = std::sqrt( (x-IC.blob_x)*(x-IC.blob_x)
              + (y-IC.blob_y)*(y-IC.blob_y)
#if (AMREX_SPACEDIM == 3)
              + (z-IC.blob_z)*(z-IC.blob_z)
#endif
              );
    //
    // Scalars, ordered as Density, Tracer(s), Temp (if using)
    //

    Real x_jump = -0.1;

    // Smooth the density interface just a bit
    scal(i,j,k,0) = //dens1 * (2. + std::tanh(100.*(-.2-x)/IC.interface_width) ) / 2.;
        IC.rho_1 + ((IC.rho_2-IC.rho_1)/2.0)*(1.0+std::tanh(-(x_jump-x)/IC.interface_width));

    if (x <= x_jump ) {
        //scal(i,j,k,0) = IC.rho_1;
        scal(i,j,k,1) = IC.tra_1;
    } else {
        //scal(i,j,k,0) = IC.rho_2;
        scal(i,j,k,1) = IC.tra_2;
    }

    for ( int nt=2; nt<nscal; nt++)
    {
        scal(i,j,k,nt) = 0.5*(1.0-std::tanh(25.*(dist-IC.blob_radius)/IC.interface_width));
        // scal(i,j,k,nt) = dist < IC.blob_radius ? 1.0 : 0.0;
        // scal(i,j,k,nt) = 0.0;
    }
  });
}

void NavierStokes::init_DoubleShearLayer (Box const& vbx,
                      Array4<Real> const& /*press*/,
                      Array4<Real> const& vel,
                      Array4<Real> const& scal,
                      const int nscal,
                      Box const& domain,
                      GpuArray<Real, AMREX_SPACEDIM> const& dx,
                      GpuArray<Real, AMREX_SPACEDIM> const& problo,
                      GpuArray<Real, AMREX_SPACEDIM> const& /*probhi*/,
                      InitialConditions IC)
{
  const auto domlo = amrex::lbound(domain);

  if ( !(IC.direction == 0 || IC.direction == 1) )
    amrex::Abort("\n    init_DoubleShearLayer: Must set a direction with prob.direction = 0 or 1\n    in the inputs file.  Shear layer along the z-direction not yet written");

  amrex::ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {
    Real x = problo[0] + (i - domlo.x + 0.5)*dx[0];
    Real y = problo[1] + (j - domlo.y + 0.5)*dx[1];
#if (AMREX_SPACEDIM == 3)
    Real z = problo[2] + (k - domlo.z + 0.5)*dx[2];
#endif

    //
    // Fill Velocity
    //
    if ( IC.direction == 1 )
    {
      // shear layer in y-dir
      vel(i,j,k,0) = -.05*std::sin(Pi*y);
      vel(i,j,k,1) = std::tanh(30.*(.5-amrex::Math::abs(x))/IC.interface_width);
    }
    else
    {
      // shear layer in x-dir
      vel(i,j,k,0) = std::tanh(30.*(.5-amrex::Math::abs(y))/IC.interface_width);
      vel(i,j,k,1) = .05*std::sin(Pi*x);
    }

    Real dist = std::sqrt( (x-IC.blob_x)*(x-IC.blob_x)
              + (y-IC.blob_y)*(y-IC.blob_y)
#if (AMREX_SPACEDIM == 3)
              + (z-IC.blob_z)*(z-IC.blob_z)
#endif
              );

    //
    // Scalars, ordered as Density, Tracer(s)
    //
    scal(i,j,k,0) = IC.density;

    // Tracers
    scal(i,j,k,1) = dist < IC.blob_radius ? 1.0 : 0.0;
    for ( int nt=2; nt<nscal; nt++)
    {
      scal(i,j,k,nt) = 1.0;
    }
  });
}

void NavierStokes::init_RayleighTaylor (Box const& vbx,
                    Array4<Real> const& /*press*/,
                    Array4<Real> const& /*vel*/,
                    Array4<Real> const& scal,
                    const int nscal,
                    Box const& domain,
                    GpuArray<Real, AMREX_SPACEDIM> const& dx,
                    GpuArray<Real, AMREX_SPACEDIM> const& problo,
                    GpuArray<Real, AMREX_SPACEDIM> const& probhi,
                    InitialConditions IC)
{
  const auto domlo = amrex::lbound(domain);

  //
  // Velocity already initialized to 0
  //

  //
  // Scalars, ordered as Density, Tracer(s), Temp (if using)
  //
  const Real Lx    = (probhi[0] - problo[0]);

#if (AMREX_SPACEDIM == 2)
  amrex::ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {
    Real x = problo[0] + (i - domlo.x + 0.5)*dx[0];
    Real y = problo[1] + (j - domlo.y + 0.5)*dx[1];

    const Real pertheight = 0.5 + IC.pertamp*(std::cos(2.0*Pi*x/Lx)
                          + std::cos(2.0*Pi*(Lx-x)/Lx));

    scal(i,j,k,0) = IC.rho_1 + ((IC.rho_2-IC.rho_1)/2.0)*(1.0+std::tanh((y-pertheight)/IC.interface_width));
    scal(i,j,k,1) = IC.tra_1 + ((IC.tra_2-IC.tra_1)/2.0)*(1.0+std::tanh((y-pertheight)/IC.interface_width));
    for ( int nt=2; nt<nscal; nt++)
    {
      scal(i,j,k,nt) = 1.0;
    }

    // // Initialize the LS function if do_phi
    // if (do_phi) {
    //   scal(i,j,k,IC.IC.phicomp-IC.Density) = y-pertheight;
    // } 

  });

#elif (AMREX_SPACEDIM == 3)

  const Real Ly    = (probhi[1] - problo[1]);
  const Real splitz = 0.5*(problo[2] + probhi[2]);

  // Create random amplitudes and phases for the perturbation

  //This doesn't work for OMP. Just hard-code results below.
  // amrex::InitRandom(111397);
  // Real rn = amrex::Random();
  // const Real ranampl = 2.*(rn-0.5);

  // rn = amrex::Random();
  // const Real ranphse1 = 2.*Pi*rn;

  // rn = amrex::Random();
  // const Real ranphse2 = 2.*Pi*rn;

  const Real ranampl = 2.*(0.6544437533747718 - 0.5);
  const Real ranphse1 = 2.*Pi*0.1556190326530211;
  const Real ranphse2 = 2.*Pi*0.4196144025537369;

  amrex::ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {
    Real x = problo[0] + (i - domlo.x + 0.5)*dx[0];
    Real y = problo[1] + (j - domlo.y + 0.5)*dx[1];
    Real z = problo[2] + (k - domlo.z + 0.5)*dx[2];

    Real pert = ranampl * std::sin(2.0*Pi*x/Lx + ranphse1 )
                        * std::sin(2.0*Pi*y/Ly + ranphse2 );

    Real pertheight = splitz - IC.pertamp*pert;

    scal(i,j,k,0) = IC.rho_1 + ((IC.rho_2-IC.rho_1)/2.0)*(1.0+std::tanh((z-pertheight)/IC.interface_width));
    scal(i,j,k,1) = IC.tra_1 + ((IC.tra_2-IC.tra_1)/2.0)*(1.0+std::tanh((z-pertheight)/IC.interface_width));
    for ( int nt=2; nt<nscal; nt++)
    {
      scal(i,j,k,nt) = 1.0;
    }
  });

#endif
}


// 
// ls related
// 
void NavierStokes::init_RayleighTaylor_LS (Box const& vbx,
                    Array4<Real> const& /*press*/,
                    Array4<Real> const& /*vel*/,
                    Array4<Real> const& scal,
                    const int nscal,
                    Box const& domain,
                    GpuArray<Real, AMREX_SPACEDIM> const& dx,
                    GpuArray<Real, AMREX_SPACEDIM> const& problo,
                    GpuArray<Real, AMREX_SPACEDIM> const& probhi,
                    InitialConditions IC)
{
  const auto domlo = amrex::lbound(domain);

  //
  // Velocity already initialized to 0
  //

  BL_ASSERT(AMREX_SPACEDIM==2);
  //
  // Scalars, ordered as Density, Tracer(s), Temp (if using), LS
  //
  const Real Lx    = (probhi[0] - problo[0]);

#if (AMREX_SPACEDIM == 2)
  amrex::ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {
    Real x = problo[0] + (i - domlo.x + 0.5)*dx[0];
    Real y = problo[1] + (j - domlo.y + 0.5)*dx[1];

    const Real pertheight = 2.0 + IC.pertamp*(std::cos(2.0*Pi*x/Lx));

    scal(i,j,k,0) = IC.rho_2 + ((IC.rho_1-IC.rho_2)/2.0)*(1.0+std::tanh((y-pertheight)/IC.interface_width));
    for ( int nt=1; nt<nscal; nt++)
    {
      scal(i,j,k,nt) = 1.0;
    }

    // Initialize the LS function if do_phi
    if (IC.do_phi) {
      scal(i,j,k,IC.phicomp-IC.Density) = y-pertheight;
    } 

  });

#elif (AMREX_SPACEDIM == 3)
#endif
}

// 
// ls related
// 
void NavierStokes::init_BreakingWave (Box const& vbx,
                    Array4<Real> const& /*press*/,
                    Array4<Real> const& vel,
                    Array4<Real> const& scal,
                    const int nscal,
                    Box const& domain,
                    GpuArray<Real, AMREX_SPACEDIM> const& dx,
                    GpuArray<Real, AMREX_SPACEDIM> const& problo,
                    GpuArray<Real, AMREX_SPACEDIM> const& probhi,
                    InitialConditions IC)
{
  const auto domlo = amrex::lbound(domain);

  BL_ASSERT(AMREX_SPACEDIM==2);
  //
  // Scalars, ordered as Density, Tracer(s), Temp (if using), LS
  //
  const Real Lx    = (probhi[0] - problo[0]);

  // wave parameters
  const Real WAVE_LENGTH = Lx;
  const Real KA = 0.55;
  const Real K_WAVE = 2.0*Pi/WAVE_LENGTH;
  const Real O_WAVE = std::sqrt((K_WAVE*9.81)*(1.0+KA*KA/2.0));
  // amrex::Print() << "IC.rho_w IC.rho_a " << IC.rho_w << " " << IC.rho_a << std::endl;

#if (AMREX_SPACEDIM == 2)
  amrex::ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {
    Real x = problo[0] + (i - domlo.x + 0.5)*dx[0];
    Real y = problo[1] + (j - domlo.y + 0.5)*dx[1];

    const Real eta =  KA/K_WAVE*std::cos(K_WAVE*x)
                      + 0.5    *std::pow(KA,2.0)/K_WAVE*std::cos(2.0*K_WAVE*x)
                      + 3.0/8.0*std::pow(KA,3.0)/K_WAVE*std::cos(3.0*K_WAVE*x);

    const Real ufs = O_WAVE*KA/K_WAVE*std::exp(K_WAVE*eta)*std::cos(K_WAVE*x);
    const Real vfs = O_WAVE*KA/K_WAVE*std::exp(K_WAVE*eta)*std::sin(K_WAVE*x);

    //
    // Fill Velocity
    //
    if ( y <= eta )
    {
      vel(i,j,k,0) = O_WAVE*KA/K_WAVE*std::exp(K_WAVE*y)*std::cos(K_WAVE*x);
      vel(i,j,k,1) = O_WAVE*KA/K_WAVE*std::exp(K_WAVE*y)*std::sin(K_WAVE*x);
      scal(i,j,k,0) = IC.rho_w;
    }
    else
    {
      vel(i,j,k,0) = ufs*std::exp(-100.0*(y-eta));
      vel(i,j,k,1) = vfs*std::exp(-100.0*(y-eta));
      scal(i,j,k,0) = IC.rho_a;
    }

    for ( int nt=1; nt<nscal; nt++)
    {
      scal(i,j,k,nt) = 1.0;
    }

    // Initialize the LS function if do_phi
    if (IC.do_phi) {
      scal(i,j,k,IC.phicomp-IC.Density) = eta-y;
    } 

  });

#elif (AMREX_SPACEDIM == 3)
#endif
}

// -----------------------------------------------------------
// This case is an unsteady viscous benchmark for which the
// exact solution in 2D is
//     u(x,y,t) =   Sin(2 Pi x) Cos(2 Pi y) Exp(-2 (2Pi)^2 Nu t)
//     v(x,y,t) = - Cos(2 Pi x) Sin(2 Pi y) Exp(-2 (2Pi)^2 Nu t)
//     p(x,y,t) = - {Cos(4 Pi x) + Cos(4 Pi y)} Exp(-4 (2Pi)^2 Nu t) / 4
// In Exec/benchmarks, there is a tool ViscBench2d.cpp that reads a
// plot file and compares the solution against this exact solution.
// This benchmark was originally derived by G.I. Taylor (Phil. Mag.,
// Vol. 46, No. 274, pp. 671-674, 1923) and Ethier and Steinman
// (Intl. J. Num. Meth. Fluids, Vol. 19, pp. 369-375, 1994) give
// the pressure field.
// For the 3D problem here, TAYLOR, G. I. & GREEN, A. E. 1937 Mechanism of the production of small eddies from large ones. Proc. R. Soc. Lond. A 158, 499–521.
// gives an approximate solution from perurbation theory.
//
// Analytic 3D solutions (not implemented here) are discussed in, for example,
// Antuono, M. (2020). Tri-periodic fully three-dimensional analytic solutions for the Navier–Stokes equations. Journal of Fluid Mechanics, 890, A23. doi:10.1017/jfm.2020.126
//
void NavierStokes::init_TaylorGreen (Box const& vbx,
                     Array4<Real> const& /*press*/,
                     Array4<Real> const& vel,
                     Array4<Real> const& scal,
                     const int nscal,
                     Box const& domain,
                     GpuArray<Real, AMREX_SPACEDIM> const& dx,
                     GpuArray<Real, AMREX_SPACEDIM> const& problo,
                     GpuArray<Real, AMREX_SPACEDIM> const& /*probhi*/,
                     InitialConditions IC)
{
  const auto domlo = amrex::lbound(domain);

  if ( IC.v_x == 0.0 )
    amrex::Abort("NavierStokes::init_TaylorGreen: Must provide prob.velocity_factor. If unsure, prob.velocity_factor = 1.0 is a good choice.");

  amrex::ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {
    Real x = problo[0] + (i - domlo.x + 0.5)*dx[0];
    Real y = problo[1] + (j - domlo.y + 0.5)*dx[1];
#if (AMREX_SPACEDIM == 3)
    Real z = problo[2] + (k - domlo.z + 0.5)*dx[2];
#else
    constexpr Real z = 0.0;
#endif

    //
    // Fill Velocity
    //
    AMREX_D_TERM(vel(i,j,k,0) =  IC.v_x*std::sin(IC.a*TwoPi*x) * std::cos(IC.b*TwoPi*y) * std::cos(IC.c*TwoPi*z);,
         vel(i,j,k,1) = -IC.v_x*std::cos(IC.a*TwoPi*x) * std::sin(IC.b*TwoPi*y) * std::cos(IC.c*TwoPi*z);,
         vel(i,j,k,2) = 0.0;);

    //
    // Scalars, ordered as Density, Tracer(s)
    //
    scal(i,j,k,0) = IC.density;

    // The theoretical pressure perturbation from p_0
#if ( AMREX_SPACEDIM == 2 )
    scal(i,j,k,1) = (IC.density*IC.v_x*IC.v_x/4.0)*(cos(2.0*IC.a*TwoPi*x)+cos(2.0*IC.b*TwoPi*y));
#else
    scal(i,j,k,1) = (IC.density*IC.v_x*IC.v_x/16.0)*(2.0+cos(2.0*IC.c*TwoPi*z))*(cos(2.0*IC.a*TwoPi*x)+cos(2.0*IC.b*TwoPi*y));
#endif

    // Tracers
    for ( int nt=2; nt<nscal; nt++)
    {
      scal(i,j,k,nt) = 1.0;
    }
  });
}

void NavierStokes::init_particles (Box const& vbx,
                     Array4<Real> const& press,
                     Array4<Real> const& vel,
                     Array4<Real> const& scal,
                     const int nscal,
                     Box const& domain,
                     GpuArray<Real, AMREX_SPACEDIM> const& dx,
                     GpuArray<Real, AMREX_SPACEDIM> const& problo,
                     GpuArray<Real, AMREX_SPACEDIM> const& /*probhi*/,
                     InitialConditions IC)
{

  const auto domlo = amrex::lbound(domain);

  amrex::ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {
    // Real x = problo[0] + (i - domlo.x + 0.5)*dx[0];
    // Real y = problo[1] + (j - domlo.y + 0.5)*dx[1];
    // Real z = problo[2] + (k - domlo.z + 0.5)*dx[2];

    //
    // Fill Velocity
    //
    vel(i,j,k,0) = 0.0;
    vel(i,j,k,1) = 0.0;

#if (AMREX_SPACEDIM == 3)
    vel(i,j,k,2) = 0.0;
#endif

    //
    // Scalars, ordered as Density, Tracer(s), Temp (if using)
    //

    // All Tracers are set here
    for ( int nt=0; nt<nscal; nt++)
    {
      scal(i,j,k,nt) = 1.0;
    }

  });
}

void NavierStokes::init_Euler (Box const& vbx,
                   Array4<Real> const& /*press*/,
                   Array4<Real> const& vel,
                   Array4<Real> const& scal,
                   const int nscal,
                   Box const& domain,
                   GpuArray<Real, AMREX_SPACEDIM> const& dx,
                   GpuArray<Real, AMREX_SPACEDIM> const& problo,
                   GpuArray<Real, AMREX_SPACEDIM> const& /*probhi*/,
                   InitialConditions IC)
{
  const auto domlo = amrex::lbound(domain);

#if (AMREX_SPACEDIM != 3)
    amrex::Abort("NavierStokes::init_Euler: This is a 3D problem, please recompile with DIM=3 in makefile");
#endif

  constexpr Real eps_input=0.05, rho_input=0.15;
  constexpr Real beta_input=15.0, delta_input=0.0333;
  constexpr Real kappa_input=500.0;

  amrex::ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {
    Real x = problo[0] + (i - domlo.x + 0.5)*dx[0] - 0.5;
    Real y = problo[1] + (j - domlo.y + 0.5)*dx[1] - 0.5;
    Real z = problo[2] + (k - domlo.z + 0.5)*dx[2] - 0.5;

    Real r_yz = std::sqrt(y*y+z*z);

    //
    // Fill Velocity
    //
    vel(i,j,k,0) = tanh( (rho_input - r_yz) / delta_input);
    vel(i,j,k,1) = 0.0;
    vel(i,j,k,2) = eps_input * std::exp(-beta_input * (x*x + y*y) );

    //
    // Scalars, ordered as Density, Tracer(s)
    //
    scal(i,j,k,0) = IC.density;
    scal(i,j,k,1) = std::exp( -kappa_input * (rho_input - r_yz)*(rho_input - r_yz) );

    // Additional Tracer, if using
    for ( int nt=2; nt<nscal; nt++)
    {
      scal(i,j,k,nt) = 1.0;
    }
  });
}

//
// Euler vortex in isentropic flow. Analytic solution is translation of the
// initial conditions based on propagation speed and simulation time.
// There are many references, for example
// Spiegel, Seth & Huynh, H.T. & DeBonis, James. (2015). A Survey of the Isentropic Euler Vortex Problem using High-Order Methods. 10.2514/6.2015-2444.
//
void NavierStokes::init_ConvectedVortex (Box const& vbx,
                                         Array4<Real> const& /*press*/,
                                         Array4<Real> const& vel,
                                         Array4<Real> const& scal,
                                         const int nscal,
                                         Box const& domain,
                                         GpuArray<Real, AMREX_SPACEDIM> const& dx,
                                         GpuArray<Real, AMREX_SPACEDIM> const& problo,
                                         GpuArray<Real, AMREX_SPACEDIM> const& /*probhi*/,
                                         InitialConditions IC)
{
  const auto domlo = amrex::lbound(domain);

  amrex::ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {
    AMREX_D_TERM(Real x = problo[0] + (i - domlo.x + 0.5)*dx[0];,
                 Real y = problo[1] + (j - domlo.y + 0.5)*dx[1];,
                 /*Real z = problo[2] + (k - domlo.z + 0.5)*dx[2]*/);

    amrex::Real deltax = x - IC.a; // x-distance from vortex center
    amrex::Real deltay = y - IC.b; // y-distance form vortex center
    amrex::Real d_sq = deltax*deltax + deltay*deltay;
    amrex::Real r_sq = IC.c * IC.c; // square of vortex radius
    amrex::Real u_vort = -IC.forcevort*deltay/r_sq * exp(-d_sq/r_sq/2.);
    amrex::Real v_vort = IC.forcevort*deltax/r_sq * exp(-d_sq/r_sq/2.);
#if (AMREX_SPACEDIM == 3)
    amrex::Real w_vort = 0.;
#endif

    //
    // Fill Velocity
    //
    switch(IC.meanFlowDir) {
      case 1  :
         AMREX_D_TERM(vel(i,j,k,0) = IC.meanFlowMag + u_vort;,
                      vel(i,j,k,1) = v_vort;,
                      vel(i,j,k,2) = w_vort);
         break;
      case -1 :
         AMREX_D_TERM(vel(i,j,k,0) = -IC.meanFlowMag + u_vort;,
                      vel(i,j,k,1) = v_vort;,
                      vel(i,j,k,2) = w_vort);
         break;
      case 2 :
         AMREX_D_TERM(vel(i,j,k,0) = v_vort;,
                      vel(i,j,k,1) = IC.meanFlowMag + u_vort;,
                      vel(i,j,k,2) = w_vort);
         break;
      case -2 :
         AMREX_D_TERM(vel(i,j,k,0) = v_vort;,
                      vel(i,j,k,1) = -IC.meanFlowMag + u_vort;,
                      vel(i,j,k,2) = w_vort);
         break;
      case 3 :
         AMREX_D_TERM(vel(i,j,k,0) = IC.meanFlowMag + u_vort;,
                      vel(i,j,k,1) = IC.meanFlowMag + v_vort;,
                      vel(i,j,k,2) = w_vort);
         break;
      case -3 :
         AMREX_D_TERM(vel(i,j,k,0) = -IC.meanFlowMag + u_vort;,
                      vel(i,j,k,1) = -IC.meanFlowMag + v_vort;,
                      vel(i,j,k,2) = w_vort);
         break;
    }

    //
    // Scalars, ordered as Density, Tracer(s)
    //
    scal(i,j,k,0) = IC.density;

    // Additional Tracer, if using
    for ( int nt=1; nt<nscal; nt++)
    {
      scal(i,j,k,nt) = 1.0;
    }
  });
}
