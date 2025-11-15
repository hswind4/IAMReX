//
// Created by aoe on 25-7-7.
//

#ifndef DIFFUSEDIB_PARALLEL_H
#define DIFFUSEDIB_PARALLEL_H

#include <AMReX_Particles.H>
#include <AMReX_MultiFabUtil.H>

#include <AMReX_RealVect.H>
#include "Collision.H"

using namespace amrex;

/**
 * @defgroup DFIBM direct forcing immersed boundary method
 *
 * IBM class
 */

/// @{

/**
 * @name useful function
 */

/// @{

AMREX_INLINE AMREX_GPU_DEVICE
/**
 * Convert nodal level set function value to Heaviside function value.
 * @param phi Level set function value at node
 * @return Heaviside function value (0.0 if phi <= 0, 1.0 if phi > 0)
 */
Real nodal_phi_to_heavi(Real phi);

/**
 * Convert nodal level set function to particle volume fraction (PVF).
 * @param pvf Output MultiFab containing particle volume fraction
 * @param phi_nodal Input MultiFab containing nodal level set function
 */
void nodal_phi_to_pvf(MultiFab& pvf, const MultiFab& phi_nodal);

/**
 * Compute delta function value for immersed boundary method.
 * @param xf Eulerian grid point coordinate
 * @param xp Lagrangian marker coordinate
 * @param h Grid spacing
 * @param value Output delta function value
 */
void deltaFunction(Real xf, Real xp, Real h, Real& value);

/// @}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                     particle and markers                      */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/// @cond
enum P_ATTR_REAL{
    U_Marker = 0,
    V_Marker,
    W_Marker,
    Fx_Marker,
    Fy_Marker,
    Fz_Marker,
    Mx_Marker,
    My_Marker,
    Mz_Marker,
    num_Real
};

enum P_ATTR_INT {
    M_ID = 0,
    num_Int
};

enum DELTA_FUNCTION_TYPE{
    FOUR_POINT_IB = 0,
    THREE_POINT_IB
};

typedef struct {
    IntVect index;
    Real weight;
    Real Vcell;
    Real eps;
}MAP_INFO;

struct kernel{
    int id;
    RealVect velocity{0.0,0.0,0.0};
    RealVect location{0.0,0.0,0.0};
    RealVect omega{0.0,0.0,0.0};

    RealVect velocity_old{0.0,0.0,0.0};
    RealVect location_old{0.0,0.0,0.0};
    RealVect omega_old{0.0,0.0,0.0};

    RealVect varphi{0.0,0.0,0.0};
    Real radius{0.0};
    Real radius2{0.0};
    Real radius3{0.0};
    int geometry_type{1};  // 1 = sphere, 2 = ellipsoid, 3 = cylinder
    Real rho{0.0};
    int ml{0};
    int start_id{0};
    Real dv{0.0};
    Real Vp{0.0};
    RealVect ib_force{0.0,0.0,0.0};
    RealVect ib_moment{0.0, 0.0, 0.0};

    RealVect sum_u_new{0.0,0.0,0.0};
    RealVect sum_u_old{0.0,0.0,0.0};
    RealVect sum_t_new{0.0,0.0,0.0};
    RealVect sum_t_old{0.0,0.0,0.0};

    RealVect Fcp{0.0,0.0,0.0};
    RealVect Tcp{0.0,0.0,0.0};

    Real* phiK;
    Real* thetaK;

    IntVect TL{0, 0, 0}, RL{0, 0, 0};
};

using CusParIter = ParIter<0, 0, num_Real, num_Int>;
// lagrangian marker manager
using mParticleContainer = ParticleContainer<0, 0, num_Real, num_Int>;

class mParIter : public CusParIter{
public:

    using ParIter<0, 0, num_Real, num_Int>::ParIter;
    using RealVector = CusParIter::ContainerType::RealVector;
    using IntVector = CusParIter::ContainerType::IntVector;

    [[nodiscard]] const std::array<RealVector, num_Real>& GetAttribs () const {
        return GetStructOfArrays().GetRealData();
    }

    [[nodiscard]] const RealVector& GetAttribs (int comp) const {
        return GetStructOfArrays().GetRealData(comp);
    }

    [[nodiscard]] const IntVector& GetIDs() const {
        return GetStructOfArrays().GetIntData(M_ID);
    }

    std::array<RealVector, num_Real>& GetAttribs () {
        return GetStructOfArrays().GetRealData();
    }

    RealVector& GetAttribs (int comp) {
        return GetStructOfArrays().GetRealData(comp);
    }
};
/// @endcond

/**
 * @name Particle Manager
 *
 * @brief particle manager for AMReX particle container
 */

/// @{

class mParticle
{
public:
    explicit mParticle() = default;

    /**
     * Initialize particles with given properties and parameters.
     * @param x Vector of x-coordinates for particle centers
     * @param y Vector of y-coordinates for particle centers
     * @param z Vector of z-coordinates for particle centers
     * @param rho_s Vector of solid particle densities
     * @param Vx Vector of initial x-velocities
     * @param Vy Vector of initial y-velocities
     * @param Vz Vector of initial z-velocities
     * @param Ox Vector of initial x-angular velocities
     * @param Oy Vector of initial y-angular velocities
     * @param Oz Vector of initial z-angular velocities
     * @param TLX Vector of translational constraint flags in x-direction
     * @param TLY Vector of translational constraint flags in y-direction
     * @param TLZ Vector of translational constraint flags in z-direction
     * @param RLX Vector of rotational constraint flags in x-direction
     * @param RLY Vector of rotational constraint flags in y-direction
     * @param RLZ Vector of rotational constraint flags in z-direction
     * @param radius Vector of particle radii
     * @param h Grid spacing
     * @param gravity Gravitational acceleration
     * @param verbose Verbosity level for output
     */
    void InitParticles(const Vector<Real>& x,
                       const Vector<Real>& y,
                       const Vector<Real>& z,
                       const Vector<Real>& rho_s,
                       const Vector<Real>& Vx,
                       const Vector<Real>& Vy,
                       const Vector<Real>& Vz,
                       const Vector<Real>& Ox,
                       const Vector<Real>& Oy,
                       const Vector<Real>& Oz,
                       const Vector<int>& TLX,
                       const Vector<int>& TLY,
                       const Vector<int>& TLZ,
                       const Vector<int>& RLX,
                       const Vector<int>& RLY,
                       const Vector<int>& RLZ,
                       const Vector<Real>& radius,
                       const Vector<Real>& radius2,
                       const Vector<Real>& radius3,
                       const Vector<int>& geometry_type,
                       Real h,
                       Real gravity,
                       int verbose = 0);

    /**
     * Perform interaction between Lagrangian particles and Eulerian fluid.
     * This function handles the immersed boundary method coupling between
     * solid particles and the surrounding fluid.
     * @param EulerVel Eulerian velocity field
     * @param EulerForce Eulerian force field (output)
     * @param dt Time step size
     */
    void InteractWithEuler(MultiFab &EulerVel, MultiFab &EulerForce, Real dt = 0.1);

    /**
     * Write particle data to file for visualization or restart.
     * @param index File index for output
     */
    void WriteParticleFile(int index);

    /**
     * Update Lagrangian marker positions and properties.
     */
    void UpdateLagrangianMarker();

    /**
     * Interpolate Eulerian velocity to Lagrangian markers.
     * @param Euler Eulerian velocity field
     * @param type Velocity component type (0=x, 1=y, 2=z)
     */
    void VelocityInterpolation(amrex::MultiFab &Euler, int type);

    /**
     * Compute Lagrangian forces on immersed boundary markers.
     * @param dt Time step size
     */
    void ComputeLagrangianForce(Real dt);

    /**
     * Spread Lagrangian forces to Eulerian grid.
     * @param Euler Eulerian field for force spreading
     * @param type Force component type (0=x, 1=y, 2=z)
     */
    void ForceSpreading(amrex::MultiFab &Euler, int type);

    /**
     * Apply velocity correction for immersed boundary method.
     * @param Euler Eulerian velocity field
     * @param EulerForce Eulerian force field
     * @param dt Time step size
     */
    void VelocityCorrection(amrex::MultiFab &Euler, amrex::MultiFab &EulerForce, Real dt) const;

    /**
     * Update particle positions and properties based on fluid flow.
     * @param iStep Current time step index
     * @param time Current simulation time
     * @param Euler_old Eulerian velocity field at previous time
     * @param Euler Eulerian velocity field at current time
     * @param phi_nodal Nodal level set function
     * @param pvf Particle volume fraction
     * @param dt Time step size
     */
    void UpdateParticles(int iStep, Real time, const MultiFab& Euler_old, const MultiFab& Euler, MultiFab& phi_nodal, MultiFab& pvf, Real dt);

    /**
     * Handle particle-particle collision interactions.
     * @param model Collision model identifier
     */
    void DoParticleCollision(int model);

    static void WriteIBForceAndMoment(int step, amrex::Real time, amrex::Real dt, kernel& current_kernel);

    /**
     * Record old values for time integration.
     * @param kernel Kernel object containing particle data
     */
    void RecordOldValue(kernel& kernel);

    /**
     * Resolve Lagrangian markers from file.
     * @param marker_file File containing marker data
     */
    void ResolveLagrangianMarker(std::string marker_file);

    /**
     * Resolve particle data using RKPM (Reproducing Kernel Particle Method).
     * @param RKPM_file File containing RKPM data
     */
    void ResolveWithRPKM(std::string RKPM_file);

    /**
     * Get starting index of Lagrangian markers for a particle.
     * @param index Particle index
     * @return Starting index of Lagrangian markers
     */
    int StartOfLagrangianMarker(size_t index);

    /**
     * Get number of Lagrangian markers for a particle.
     * @param index Particle index
     * @return Number of Lagrangian markers
     */
    int NumOfLagrangianMarker(size_t index);

    /**
     * Get position of a specific Lagrangian marker.
     * @param index Marker index
     * @return Position vector of the marker
     */
    RealVect GetPositionOfMarker(size_t index);

    Vector<kernel> particle_kernels;

    mParticleContainer *mContainer{nullptr};

    ParticleCollision m_Collision;

    Vector<RealVect> LargrangianMarker;

    Vector<size_t> StartOfMarker;

    Vector<size_t> NumOfMarker;

    std::map<int, Vector<MAP_INFO>> RKPM_MAP;

    int max_largrangian_num = 0;

    uint32_t ib_force_file_index = 0;

    RealVect m_gravity{0.0,0.0,0.0};

    int verbose = 0;

    Real spend_time;
    // read write
    bool do_RKPM{false};
};

/// @}
/// @cond

class Particles{
public:
    static void create_particles(const Geometry &gm,
                                 const DistributionMapping & dm,
                                 const BoxArray & ba);

    static mParticle* get_particles();
    static void init_particle(Real gravity, Real h);
    static void Restart(Real gravity, Real h, int iStep);
    static void Initialize();
    static int ParticleFinestLevel();;

    inline static bool isInitial{false};
private:
    inline static mParticle* particle = nullptr;
};

/// @endcond
/// @}

#endif //DIFFUSEDIB_PARALLEL_H
