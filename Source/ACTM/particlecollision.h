#include<cstdio>
#include<cmath>
#include "VECTOR.H"
#include<vector>
#include<sstream>
#include<iomanip>
#include "CONSTANT.H"
using namespace std;
#ifndef Particle_Collision_H
#define Particle_Collision_H
class Particle;
class collision;
class ACTM;
VECTOR CalculateRelativeV(const Particle& a, const Particle& b); //计算接触点相对速度 A.8a
class BoundaryBox {
public:
    // 边界的结构体定义
    struct Boundary {
        bool exists; // 边界是否存在
        double coordinate; // 边界的坐标位置

        Boundary(bool e = false, double c = 0.0) : exists(e), coordinate(c) {}
    };

    // 三维区域的六个边界
    Boundary left, right, bottom, top, front, back;
    // 设置边界的函数
    void setBoundary(bool leftExists, double leftCoord,
                     bool rightExists, double rightCoord,
                     bool bottomExists, double bottomCoord,
                     bool topExists, double topCoord,
                     bool frontExists, double frontCoord,
                     bool backExists, double backCoord);
    // 计算点到边界的距离
};

class collision{
public:
    double e_t,e_n,delta_t,Tc,DensityofFluid,miu;
    VECTOR gravity; //重力加速度矢量
    int step;
    int nowstep=0;
    llu nparticles;
    bool isco=false;//用来标记是否碰上了的，主要是看回弹速度
    bool endflag=false;//用途同上
    int stepseparate=-1;
    //同上
    double proportion;
    vector<Particle>MyParticles;
    vector<vector<ACTM>>MyACTMs;//
    vector<vector<VECTOR>>RelativeTanDistance;//切向相对位移
    BoundaryBox box;

    stringstream ss;
    collision( double _e_t, double _e_n, double _delta_t, double _gravity,double _miu, double _DensityofFluid);
    void setBoundary(BoundaryBox);
    void setstep(int);//设置时间步长
    void setnparticles(llu);//设置小球个数
    void initializeparticles();//给所有小球赋初值,考虑可能会需要给每个小球改变半径或者初始坐标，可以用vector类型的形参
    void initializeACTM(llu); //初始化软球模型
    void CauculateCoefACTM();//计算软球模型的系数
    void CalculateForceandMoment();//计算力和力矩
    void ParticleStatusChange();
    void output();
};


class Particle
{
private:
    /* data */
public:
    VECTOR X;//位移
    VECTOR V;//速度
    VECTOR a;//加速度
    VECTOR omega;//角速度

    VECTOR alpha;//角加速度
    VECTOR force;//收到的外力
    VECTOR moment;//力矩
    double density;
    double radius;
    double volume;
    double mass;
    double inertia;//惯量
    //Particle(/* args */);
   // ~Particle();
};

class ACTM {
public:
    double k_n;
    double d_n;
    double k_t;
    double d_t;

    // 默认构造函数
    ACTM() : k_n(0.0), d_n(0.0), k_t(0.0), d_t(0.0) {}
};

#endif