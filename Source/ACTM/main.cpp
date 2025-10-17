#include<cstdio>
#include<cmath>
#include "particlecollision.h"
#include<iostream>
#include<ctime>
#include<sstream>
#include<fstream>
#include"VECTOR.H"
using namespace std;

void writeToFile(const std::string& filePath, const std::stringstream& dataStream) {  
    std::ofstream file(filePath);  
    if (file.is_open()) {  
        file << dataStream.rdbuf(); // 将字符串流的内容写入文件  
        file.close();  
    } else {  
        std::cerr << "Unable to open file for writing." << std::endl;  
    }  
} 
//初始化常数设置信息 y
//定义了一个矢量类 y 乘法待重载
//定义一个碰撞模型ACTM类最后搞一个类类型的二维数组 y
//定义一个初始常数类 y
//定义一个单个小球状态类，位移（坐标），速度，加速度 y
//定义一个流场边界类 
//判断小球是否和壁面（小球）发生了碰撞，润滑  估计也就两三行的一个函数
//方程右端的力的计算 需要判断力的类型，然后累加 
//单步欧拉推进
//待完成：initializeparticles中小球信息初始化+CalculateForceandMoment中IB力的计算还有力矩方程的右端
//可能出错，力的方向，切向力有个/Ksi_t? kt的计算
int main()
{
    clock_t start_t,end_t;
    start_t=clock();
    //            时间步长      恢复系数            重力加速度   摩擦因数 
    const double delta_t=0.000001,e_n=0.97,e_t=0.34,gravity=10*9.81,densityofFLUID=998,miu=0.11;
    llu nparticles;
    int steps=12000;
    nparticles=1;
    collision Problem(e_t,e_n,delta_t,gravity,miu,densityofFLUID);
    BoundaryBox box;  //区域（边界）信息 
    // 设置边界框的边界  
    box.setBoundary(true, 0.0,   // 左边界  
                    true, 1.0,  // 右边界  
                    true, 0.0,   // 底边界  
                    true, 1.0,  // 顶边界  
                    true, 0.0,   // 前边界  
                    true, 1.0); // 后边界          
    
    Problem.setstep(steps); //设置推进步数
    Problem.setBoundary(box);
    Problem.setnparticles(nparticles); //申请内存    
    Problem.initializeparticles();//目前只进行了坐标赋值,密度和半径都是默认的
    Problem.initializeACTM(nparticles);
    Problem.CauculateCoefACTM();
    for(int k=1;k<=100;k++){//更换不同的初始自转角速度，以验证切向力模型
   // Problem.ss<<k<<endl;
    Problem.nowstep=0;
    Problem.initializeparticles();
    Problem.MyParticles[0].omega=VECTOR(0,-100.0*k,0);//初始加旋转
  //  Problem.ss<<k*0.0125<<" ";
    for(int i=0;i<steps;i++){  
    Problem.CalculateForceandMoment(); 
    Problem.ParticleStatusChange();
    Problem.output();
    }
    }
    writeToFile("height.dat",Problem.ss);
    end_t=clock();
    cout<<"RunTime:"<<double(end_t-start_t)/CLOCKS_PER_SEC<<'s'<<endl;
    return 0;
}