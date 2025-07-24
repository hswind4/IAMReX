//
// Created by aoe on 25-7-7.
//

#ifndef DIFFUSEDIB_PARALLEL_H
#define DIFFUSEDIB_PARALLEL_H

#include <AMReX_Particles.H>
#include <AMReX_MultiFabUtil.H>

#include <AMReX_RealVect.H>
#include "Collision.H"
// using deltaFuncType = std::function<AMREX_GPU_HOST_DEVICE void(Real, Real, Real, Real&)>;

using namespace amrex;

AMREX_INLINE AMREX_GPU_DEVICE
Real nodal_phi_to_heavi(Real phi);

void nodal_phi_to_pvf(MultiFab& pvf, const MultiFab& phi_nodal);

void deltaFunction(Real xf, Real xp, Real h, Real& value);

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                     particle and markers                      */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

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

/**
 * particle information
 */
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

    Gpu::DeviceVector<Real> phiK;
    Gpu::DeviceVector<Real> thetaK;

    IntVect TL{0, 0, 0}, RL{0, 0, 0};
};

using CusParIter = ParIter<0, 0, num_Real, num_Int>;
// lagrangian marker manager
using mParticleContainer = ParticleContainer<0, 0, num_Real, num_Int>;

template<typename T, int size = 6>
class ParallelVector {
public:
    ParallelVector() = default;

    ParallelVector& operator +=(const ParallelVector& v) {
        for (int i = 0; i < size; i++) {
            data[i] += v[i];
        }
        return *this;
    }

    ParallelVector& operator =(int v) {
        for (int i = 0; i < size; i++) {
            data[i] = v;
        }
        return *this;
    }

    T at(const int index) {
        return data.at(index);
    }

    void setValue(const int index, const T& d) {
        data[index] = d;
    }

private:
    std::array<T, size> data = {};
};

/**
 * lagrangian marker iterator
 */
class mParIter : public CusParIter{
public:

    using ParIter<0, 0, num_Real, num_Int>::ParIter;
    using RealVector = CusParIter::ContainerType::RealVector;
    using IntVector = CusParIter::ContainerType::IntVector;

    /**
     *
     * @return const particle real datas
     */
    [[nodiscard]] const std::array<RealVector, num_Real>& GetAttribs () const {
        return GetStructOfArrays().GetRealData();
    }

    /**
     *
     * @param comp attribute index
     * @return const particle index attribute
     */
    [[nodiscard]] const RealVector& GetAttribs (int comp) const {
        return GetStructOfArrays().GetRealData(comp);
    }

    /**
     *
     * @return particle's id
     */
    [[nodiscard]] const IntVector& GetIDs() const {
        return GetStructOfArrays().GetIntData(M_ID);
    }

    /**
     *
     * @return particle real datas non-const
     */
    std::array<RealVector, num_Real>& GetAttribs () {
        return GetStructOfArrays().GetRealData();
    }

    /**
     *
     * @param comp attribute index
     * @return particle index real data
     */
    RealVector& GetAttribs (int comp) {
        return GetStructOfArrays().GetRealData(comp);
    }
};

/**
 * particle manager
 */
class mParticle
{
public:
    explicit mParticle() = default;

    /**
     *
     * @param x
     * @param y
     * @param z
     * @param rho_s
     * @param Vx
     * @param Vy
     * @param Vz
     * @param Ox
     * @param Oy
     * @param Oz
     * @param TLX
     * @param TLY
     * @param TLZ
     * @param RLX
     * @param RLY
     * @param RLZ
     * @param radius
     * @param h
     * @param gravity
     * @param verbose
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
                       Real h,
                       Real gravity,
                       int verbose = 0);

    /**
     *
     * @param EulerVel
     * @param EulerForce
     * @param dt
     * @param type
     */
    void InteractWithEuler(MultiFab &EulerVel, MultiFab &EulerForce, Real dt = 0.1, DELTA_FUNCTION_TYPE type = FOUR_POINT_IB);

    /**
     *
     * @param index
     */
    void WriteParticleFile(int index);

    void UpdateLagrangianMarker();

    /**
     *
     * @param Euler
     * @param type
     */
    void VelocityInterpolation(amrex::MultiFab &Euler, DELTA_FUNCTION_TYPE type);

    /**
     * compute euler force
     * @param dt time step
     */
    void ComputeLagrangianForce(Real dt);

    /**
     *
     * @param Euler Fluid MF
     * @param type  Delta function select
     */
    void ForceSpreading(amrex::MultiFab &Euler, DELTA_FUNCTION_TYPE type);

    /**
     *
     * @param Euler Fluid MF
     * @param EulerForce Euler force MF
     * @param dt Time step
     */
    void VelocityCorrection(amrex::MultiFab &Euler, amrex::MultiFab &EulerForce, Real dt) const;

    /**
     *
     * @param iStep steps
     * @param time time
     * @param Euler_old Fluid MF in last time step
     * @param Euler Fluid MF in current time step
     * @param phi_nodal phi function
     * @param pvf particle volume fraction
     * @param dt time step
     */
    void UpdateParticles(int iStep, Real time, const MultiFab& Euler_old, const MultiFab& Euler, MultiFab& phi_nodal, MultiFab& pvf, Real dt);

    /**
     *
     * @param model collision model
     */
    void DoParticleCollision(int model);

    /**
     *
     * @param step istep
     * @param time
     * @param dt time step
     * @param current_kernel particle data
     */
    static void WriteIBForceAndMoment(int step, amrex::Real time, amrex::Real dt, kernel& current_kernel);

    /**
     *
     * @param kernel particle
     */
    void RecordOldValue(kernel& kernel);

    Vector<kernel> particle_kernels;

    mParticleContainer *mContainer{nullptr};

    ParticleCollision m_Collision;

    int max_largrangian_num = 0;

    uint32_t ib_force_file_index = 0;

    RealVect m_gravity{0.0,0.0,0.0};

    int verbose = 0;

    Real spend_time;
};

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



#endif //DIFFUSEDIB_PARALLEL_H
