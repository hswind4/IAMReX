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

class mParticle
{
public:
    explicit mParticle() = default;

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

    void InteractWithEuler(MultiFab &EulerVel, MultiFab &EulerForce, Real dt = 0.1);

    void WriteParticleFile(int index);

    void UpdateLagrangianMarker();

    void VelocityInterpolation(amrex::MultiFab &Euler, int type);

    void ComputeLagrangianForce(Real dt);

    void ForceSpreading(amrex::MultiFab &Euler, int type);

    void VelocityCorrection(amrex::MultiFab &Euler, amrex::MultiFab &EulerForce, Real dt) const;

    void UpdateParticles(int iStep, Real time, const MultiFab& Euler_old, const MultiFab& Euler, MultiFab& phi_nodal, MultiFab& pvf, Real dt);

    void DoParticleCollision(int model);

    static void WriteIBForceAndMoment(int step, amrex::Real time, amrex::Real dt, kernel& current_kernel);

    void RecordOldValue(kernel& kernel);

    void ResolveLagrangianMarker(std::string marker_file);

    void ResolveWithRPKM(std::string RKPM_file);

    int StartOfLagrangianMarker(size_t index);

    int NumOfLagrangianMarker(size_t index);

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
