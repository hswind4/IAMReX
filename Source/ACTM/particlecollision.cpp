#include "particlecollision.h"
collision::collision(double _e_t, double _e_n, double _delta_t,double _gravity,double _miu, double _DensityofFluid)
{
    //设置碰撞模型的物性参数
    this->delta_t=_delta_t;
    this->e_n=_e_n;
    this->e_t=_e_t;
    Tc=10*this->delta_t;
    gravity=_gravity*VECTOR(0,0,-1);
    DensityofFluid=_DensityofFluid;
    miu=_miu;
    printf("et,en,dt,Tc,g = %.3f %.3f %f %f %.3f\n",this->e_t,this->e_n,this->delta_t,Tc,_gravity);
}

void collision::setBoundary(BoundaryBox _box)
{
    box=_box;
}

void collision::setstep(int _step)
{
    step=_step;
}

void collision::setnparticles(llu _nparticles)
{
    nparticles=_nparticles;
    MyParticles.resize(_nparticles);
    RelativeTanDistance.resize(_nparticles,vector<VECTOR>(_nparticles+6)); //6个墙壁
}

void collision::initializeparticles()
{
    for(llu i=0;i<nparticles;i++)
    {
        MyParticles[i].X=VECTOR(0.0005,0.0005,0.000125*46.5);
        MyParticles[i].a=VECTOR(0,0,-9.81*10);
        MyParticles[i].V=VECTOR(0,0,0);
        MyParticles[i].density=7800;
        MyParticles[i].radius=0.000125;
        MyParticles[i].omega=VECTOR(0,0,0);
        /*
        在这里初始化初始位置，半径，密度
        
                
        */
        double radius = MyParticles[i].radius; 
        MyParticles[i].volume = PAI * 4.0 / 3.0 * radius * radius * radius; 
  
        // 计算质量  
        double density = MyParticles[i].density; 
        MyParticles[i].mass = density * MyParticles[i].volume; 
        MyParticles[i].inertia=8*PAI*density*radius*radius*radius*radius*radius/15.0;
    }
}

void collision::initializeACTM(llu _nparticles)
{

    MyACTMs.resize(_nparticles,vector<ACTM>(_nparticles+1));
}

void collision::CauculateCoefACTM()
{ 
    // 计算ACTM参数
    // 多出来那一维度是用来储存和壁面的碰撞系数，
    // 其实把和壁面那一维省略掉用小球和自己的碰撞系数代替也可
    double M_eff;//effective mass
    for(llu i=0;i<nparticles;i++)
    {
        for(llu j=0;j<=nparticles;j++)
        {
            if(i==j) continue;
            if(j==nparticles) //处理壁面
                M_eff=MyParticles[i].mass;
            else
            {
                double m_p,m_q;
                m_p=MyParticles[i].mass;
                m_q=MyParticles[j].mass;
                M_eff=m_p*m_q/(m_p+m_q);

            }
            MyACTMs[i][j].k_n=M_eff*(PAI*PAI+log(e_n)*log(e_n))/(Tc*Tc);
            MyACTMs[i][j].k_t=2.0/7.0*M_eff*(PAI*PAI+log(e_t)*log(e_t))/(Tc*Tc); //存疑
            MyACTMs[i][j].d_n=-2*M_eff*log(e_n)/Tc;
            MyACTMs[i][j].d_t=-4.0/7.0*M_eff*log(e_t)/Tc;
        }
    }
}

void collision::CalculateForceandMoment()
{
    llu i,j;
    double volume,mass,radius;
    VECTOR X,V,omega;//赋值
    VECTOR M_cp(0,0,0),M_IB;//转动和平动方程的右边，累加
    double distance;//赋值
    VECTOR g;//relative velocity between particle center，赋值
    VECTOR g_cp;//relative velocity between contact point，赋值
    VECTOR g_ncp,g_tcp;//normal,tangental component，赋值
    VECTOR UnitNormalVector;//从球心指向对方，赋值
    VECTOR UnitTangentVector;//切向力方向的单位向量，赋值
    VECTOR F_g;//重力,赋值
    VECTOR F_f;//浮力，赋值    
    VECTOR F_IB;//IB力，赋值
    VECTOR F_tls;//弹簧振子系统提供的切向力,赋值
    VECTOR oldRelativeDistance;//ksi_k-1，赋值
    VECTOR newRelativeDistance;//Ksi_k，赋值
    VECTOR F_ncpthis;
    VECTOR F_tcpthis;
    for(i=0;i<nparticles;i++)
    {
        VECTOR F_ncp;//这个小球会受到的法向碰撞力，（与不同小球的碰撞力累加起来的）
        VECTOR F_tcp;//切向碰撞力
        F_ncp=VECTOR(0,0,0);
        F_tcp=VECTOR(0,0,0);
        M_cp=VECTOR(0,0,0);   
        M_IB=VECTOR(0,0,0);             
        volume = MyParticles[i].volume;
        mass = MyParticles[i].mass;
        radius = MyParticles[i].radius;
        omega = MyParticles[i].omega;        
        X=MyParticles[i].X;
        V=MyParticles[i].V;       
        F_g=mass*gravity; //受到重力
        F_f=-DensityofFluid*gravity*volume;
        //F=F-    计算浮力
        /*
        计算IB力，和IB力矩

        */
       //NOTE: 计算球之间的碰撞力              
       for(j=0;j<nparticles;j++)//检验i球和j球的距离
       {        
            if(i==j) continue;
            distance=abs(X-MyParticles[j].X)-radius-MyParticles[j].radius;
            if(distance<0.0)
            {
                UnitNormalVector=(MyParticles[j].X-X).Unit();//高危行径 方向问题
                g=V-MyParticles[j].V;
                g_cp=CalculateRelativeV(MyParticles[i],MyParticles[j]);
                g_ncp=(g_cp*UnitNormalVector)*UnitNormalVector;
                g_tcp=g_cp-g_ncp;
                //先算法向力
                F_ncpthis=(-MyACTMs[i][j].k_n*abs(distance)*UnitNormalVector \
                    -MyACTMs[i][j].d_n*g_ncp);
                F_ncp=F_ncp+F_ncpthis;
                //处理切向力，存疑,滑动摩擦和Ft
                //先求出相对位移
                oldRelativeDistance=RelativeTanDistance[i][j];
                newRelativeDistance=RelativeTanDistance[i][j]-\
                (RelativeTanDistance[i][j]*UnitNormalVector)*UnitNormalVector;
                if(newRelativeDistance!=VECTOR(0,0,0))
                    newRelativeDistance=abs(oldRelativeDistance)/abs(newRelativeDistance)*newRelativeDistance;
                newRelativeDistance=newRelativeDistance+delta_t*g_tcp;
                //然后求出弹簧振子力  
                F_tls=(-MyACTMs[i][j].k_t*newRelativeDistance \
                    -MyACTMs[i][j].d_t*g_tcp);
                UnitTangentVector=F_tls.Unit();
                //然后比较和滑动摩擦力的大小
                F_tcpthis=min(abs(F_tls),abs(miu*F_ncpthis))*UnitTangentVector;
                F_tcp=F_tcp+F_tcpthis;
                //然后更新相对位移；
                if(abs(F_tls)<abs(miu*F_ncpthis)) RelativeTanDistance[i][j]=newRelativeDistance;
                else RelativeTanDistance[i][j]=-(abs(miu*F_ncpthis)*UnitTangentVector+MyACTMs[i][j].d_t*g_tcp)*(1.0/MyACTMs[i][j].k_t);
                //处理力矩
                M_cp=M_cp+((radius*UnitNormalVector)^F_tcpthis);
            }
            else if(distance>0.0&&RelativeTanDistance[i][j]!=VECTOR(0,0,0))
            {
                RelativeTanDistance[i][j]=VECTOR(0,0,0);
            }
        /*
        润滑力
        else if(distance<    &&distance>0.0)
        */
       }
       //壁面碰撞处理
       {
            if (box.left.exists) 
            {   
                distance=abs(box.left.coordinate - X.x)-radius;
                j=nparticles;
                if(distance<0)
                {
                    
                    UnitNormalVector=VECTOR(-1,0,0);
                    g=V;
                    g_cp=g+radius*(omega ^ UnitNormalVector);
                    g_ncp=(g_cp*UnitNormalVector)*UnitNormalVector;            
                    g_tcp=g_cp-g_ncp;
                    F_ncpthis=(-MyACTMs[i][nparticles].k_n*abs(distance)*UnitNormalVector \
                    -MyACTMs[i][nparticles].d_n*(g_ncp));
                    F_ncp=F_ncp+F_ncpthis; 

                    oldRelativeDistance=RelativeTanDistance[i][j];
                    newRelativeDistance=RelativeTanDistance[i][j]-(RelativeTanDistance[i][j]*UnitNormalVector)*UnitNormalVector;
                    if(newRelativeDistance!=VECTOR(0,0,0))
                        newRelativeDistance=abs(oldRelativeDistance)/abs(newRelativeDistance)*newRelativeDistance;
                    newRelativeDistance=newRelativeDistance+delta_t*g_tcp;
                    //然后求出弹簧振子力
                    F_tls=(-MyACTMs[i][nparticles].k_t*newRelativeDistance \
                        -MyACTMs[i][nparticles].d_t*g_tcp);
                    UnitTangentVector=F_tls.Unit();
                    //然后比较和滑动摩擦力的大小
                    F_tcpthis=min(abs(F_tls),abs(miu*F_ncpthis))*UnitTangentVector;
                    F_tcp=F_tcp+F_tcpthis;
                    //然后更新相对位移；
                    if(abs(F_tls)<abs(miu*F_ncpthis)) RelativeTanDistance[i][j]=newRelativeDistance;
                    else RelativeTanDistance[i][j]=-(abs(miu*F_ncpthis)*UnitTangentVector+MyACTMs[i][nparticles].d_t*g_tcp)*(1.0/MyACTMs[i][nparticles].k_t);
                    //处理力矩
                    M_cp=M_cp+((radius*UnitNormalVector)^F_tcpthis);              
                }  
                else if(distance>0.0&&RelativeTanDistance[i][j]!=VECTOR(0,0,0))
                {
                    RelativeTanDistance[i][j]=VECTOR(0,0,0);
                }
            }
            if (box.right.exists) 
            {  
                distance=abs(box.right.coordinate - X.x)-radius;            
                j=nparticles+1;
                if(distance<0)
                {   
                    
                    UnitNormalVector=VECTOR(1,0,0);
                    g=V;
                    g_cp=g+radius*(omega ^ UnitNormalVector);
                    g_ncp=(g_cp*UnitNormalVector)*UnitNormalVector;
                    g_tcp=g_cp-g_ncp;
                    F_ncpthis=(-MyACTMs[i][nparticles].k_n*abs(distance)*UnitNormalVector \
                    -MyACTMs[i][nparticles].d_n*(g_ncp));
                    F_ncp=F_ncp+F_ncpthis;

                    oldRelativeDistance=RelativeTanDistance[i][j];
                    newRelativeDistance=RelativeTanDistance[i][j]-(RelativeTanDistance[i][j]*UnitNormalVector)*UnitNormalVector;
                    if(newRelativeDistance!=VECTOR(0,0,0))
                        newRelativeDistance=abs(oldRelativeDistance)/abs(newRelativeDistance)*newRelativeDistance;
                    newRelativeDistance=newRelativeDistance+delta_t*g_tcp;
                    //然后求出弹簧振子力
                    F_tls=(-MyACTMs[i][nparticles].k_t*newRelativeDistance \
                        -MyACTMs[i][nparticles].d_t*g_tcp);
                    UnitTangentVector=F_tls.Unit();
                    //然后比较和滑动摩擦力的大小
                    F_tcpthis=min(abs(F_tls),abs(miu*F_ncpthis))*UnitTangentVector;
                    F_tcp=F_tcp+F_tcpthis;
                    //然后更新相对位移；
                    if(abs(F_tls)<abs(miu*F_ncpthis)) RelativeTanDistance[i][j]=newRelativeDistance;
                    else RelativeTanDistance[i][j]=-(abs(miu*F_ncpthis)*UnitTangentVector+MyACTMs[i][nparticles].d_t*g_tcp)*(1.0/MyACTMs[i][nparticles].k_t);
                    //处理力矩
                    M_cp=M_cp+((radius*UnitNormalVector)^F_tcpthis); 
                }   
                else if(distance>0.0&&RelativeTanDistance[i][j]!=VECTOR(0,0,0))
                {
                    RelativeTanDistance[i][j]=VECTOR(0,0,0);
                } 
            }
            if (box.bottom.exists) 
            {              
                j=nparticles+2;
                distance=abs(box.bottom.coordinate - X.z)-radius;    
                UnitNormalVector=VECTOR(0,0,-1);
                //distance=-X*UnitNormalVector-radius;
              //  cout<<nowstep*delta_t<<" "<<X.z<<" "<<distance<<endl;       
                if(distance<0.0)
                {                   
                    UnitNormalVector=VECTOR(0,0,-1);
                    g=V;
                    g_cp=g+radius*(omega ^ UnitNormalVector);
                    g_ncp=(g_cp*UnitNormalVector)*UnitNormalVector;
                    g_tcp=g_cp-g_ncp;
                    F_ncpthis=(-MyACTMs[i][nparticles].k_n*abs(distance)*UnitNormalVector \
                    -MyACTMs[i][nparticles].d_n*(g_ncp));
                    F_ncp=F_ncp+F_ncpthis;
                    if(isco==false) {
                    isco=true;
                    stepseparate=10+nowstep;
                    }               
                    oldRelativeDistance=RelativeTanDistance[i][j];
                    newRelativeDistance=RelativeTanDistance[i][j]-(RelativeTanDistance[i][j]*UnitNormalVector)*UnitNormalVector;
                    if(newRelativeDistance!=VECTOR(0,0,0))
                        newRelativeDistance=abs(oldRelativeDistance)/abs(newRelativeDistance)*newRelativeDistance;//大小还原
                    newRelativeDistance=newRelativeDistance+delta_t*g_tcp;
                    //然后求出弹簧振子力
                    F_tls=(-MyACTMs[i][nparticles].k_t*newRelativeDistance \
                        -MyACTMs[i][nparticles].d_t*g_tcp);
                    UnitTangentVector=F_tls.Unit();
                    //然后比较和滑动摩擦力的大小
                    F_tcpthis=min(abs(F_tls),abs(miu*F_ncpthis))*UnitTangentVector;
                    F_tcp=F_tcp+F_tcpthis;
                    //然后更新相对位移；
                    if(abs(F_tls)<abs(miu*F_ncpthis)) RelativeTanDistance[i][j]=newRelativeDistance;
                    else RelativeTanDistance[i][j]=-(abs(miu*F_ncpthis)*UnitTangentVector+MyACTMs[i][nparticles].d_t*g_tcp)*(1.0/MyACTMs[i][nparticles].k_t);
                    //处理力矩
                    M_cp=M_cp+((radius*UnitNormalVector)^F_tcpthis);
                } 
                else if(distance>0.0&&RelativeTanDistance[i][j]!=VECTOR(0,0,0))
                {
                    RelativeTanDistance[i][j]=VECTOR(0,0,0);
                }  
                if(nowstep==stepseparate){
                    //cout<<MyParticles[i].V.z<<endl;
                    g=V;
                    g_cp=g+radius*(omega ^ UnitNormalVector);
                    ss<<g_cp.x/g_cp.z<<endl;
                    isco=false;
                    } 
            }
            if (box.top.exists) 
            {  
                j=nparticles+3;
                distance=abs(box.top.coordinate - X.z)-radius;            
                if(distance<0)
                {                   
                    UnitNormalVector=VECTOR(0,0,1);
                    g=V;
                    g_cp=g+radius*(omega ^ UnitNormalVector);
                    g_ncp=(g_cp*UnitNormalVector)*UnitNormalVector;
                    g_tcp=g_cp-g_ncp;
                    F_ncpthis=(-MyACTMs[i][nparticles].k_n*abs(distance)*UnitNormalVector \
                    -MyACTMs[i][nparticles].d_n*(g_ncp));
                    F_ncp=F_ncp+F_ncpthis;

                    oldRelativeDistance=RelativeTanDistance[i][j];
                    newRelativeDistance=RelativeTanDistance[i][j]-(RelativeTanDistance[i][j]*UnitNormalVector)*UnitNormalVector;
                    if(newRelativeDistance!=VECTOR(0,0,0))
                        newRelativeDistance=abs(oldRelativeDistance)/abs(newRelativeDistance)*newRelativeDistance;
                    newRelativeDistance=newRelativeDistance+delta_t*g_tcp;
                    //然后求出弹簧振子力
                    F_tls=(-MyACTMs[i][nparticles].k_t*newRelativeDistance \
                        -MyACTMs[i][nparticles].d_t*g_tcp);
                    UnitTangentVector=F_tls.Unit();
                    //然后比较和滑动摩擦力的大小
                    F_tcpthis=min(abs(F_tls),abs(miu*F_ncpthis))*UnitTangentVector;
                    F_tcp=F_tcp+F_tcpthis;
                    //然后更新相对位移；
                    if(abs(F_tls)<abs(miu*F_ncpthis)) RelativeTanDistance[i][j]=newRelativeDistance;
                    else RelativeTanDistance[i][j]=-(abs(miu*F_ncpthis)*UnitTangentVector+MyACTMs[i][nparticles].d_t*g_tcp)*(1.0/MyACTMs[i][nparticles].k_t);
                    //处理力矩
                    M_cp=M_cp+((radius*UnitNormalVector)^F_tcpthis);
                }  
                else if(distance>0.0&&RelativeTanDistance[i][j]!=VECTOR(0,0,0))
                {
                    RelativeTanDistance[i][j]=VECTOR(0,0,0);
                } 
            }
            if (box.front.exists) 
            { 
                j=nparticles+4;
                distance=abs(box.front.coordinate - X.y)-radius;
                if(distance<0)
                {
                    
                    UnitNormalVector=VECTOR(0,-1,0); 
                    g=V;
                    g_cp=g+radius*(omega ^ UnitNormalVector);
                    g_ncp=(g_cp*UnitNormalVector)*UnitNormalVector;
                    g_tcp=g_cp-g_ncp;
                    F_ncpthis=(-MyACTMs[i][nparticles].k_n*abs(distance)*UnitNormalVector \
                    -MyACTMs[i][nparticles].d_n*(g_ncp));
                    
                    F_ncp=F_ncp+F_ncpthis;

                    oldRelativeDistance=RelativeTanDistance[i][j];
                    newRelativeDistance=RelativeTanDistance[i][j]-(RelativeTanDistance[i][j]*UnitNormalVector)*UnitNormalVector;
                    if(newRelativeDistance!=VECTOR(0,0,0))
                        newRelativeDistance=abs(oldRelativeDistance)/abs(newRelativeDistance)*newRelativeDistance;
                    newRelativeDistance=newRelativeDistance+delta_t*g_tcp;
                    //然后求出弹簧振子力
                    F_tls=(-MyACTMs[i][nparticles].k_t*newRelativeDistance \
                        -MyACTMs[i][nparticles].d_t*g_tcp);
                    UnitTangentVector=F_tls.Unit();
                    //然后比较和滑动摩擦力的大小
                    F_tcpthis=min(abs(F_tls),abs(miu*F_ncpthis))*UnitTangentVector;
                    F_tcp=F_tcp+F_tcpthis;
                    //然后更新相对位移；
                    if(abs(F_tls)<abs(miu*F_ncpthis)) RelativeTanDistance[i][j]=newRelativeDistance;
                    else RelativeTanDistance[i][j]=-(abs(miu*F_ncpthis)*UnitTangentVector+MyACTMs[i][nparticles].d_t*g_tcp)*(1.0/MyACTMs[i][nparticles].k_t);
                    //处理力矩
                    M_cp=M_cp+((radius*UnitNormalVector)^F_tcpthis);
                }  
                else if(distance>0.0&&RelativeTanDistance[i][j]!=VECTOR(0,0,0))
                {
                    RelativeTanDistance[i][j]=VECTOR(0,0,0);
                } 
            }
            if (box.back.exists) 
            {  
                j=nparticles+5;
                distance=abs(box.back.coordinate - X.y)-radius;
                if(distance<0)
                {
                    UnitNormalVector=VECTOR(0,1,0);
                    g=V;
                    g_cp=g+radius*(omega ^ UnitNormalVector);
                    g_ncp=(g_cp*UnitNormalVector)*UnitNormalVector;
                    g_tcp=g_cp-g_ncp;
                    F_ncpthis=(-MyACTMs[i][nparticles].k_n*abs(distance)*UnitNormalVector \
                    -MyACTMs[i][nparticles].d_n*(g_ncp));
                    F_ncp=F_ncp+F_ncpthis;

                    oldRelativeDistance=RelativeTanDistance[i][j];
                    newRelativeDistance=RelativeTanDistance[i][j]-(RelativeTanDistance[i][j]*UnitNormalVector)*UnitNormalVector;
                    if(newRelativeDistance!=VECTOR(0,0,0))
                        newRelativeDistance=abs(oldRelativeDistance)/abs(newRelativeDistance)*newRelativeDistance;
                    newRelativeDistance=newRelativeDistance+delta_t*g_tcp;
                    //然后求出弹簧振子力
                    F_tls=(-MyACTMs[i][nparticles].k_t*newRelativeDistance \
                        -MyACTMs[i][nparticles].d_t*g_tcp);
                    UnitTangentVector=F_tls.Unit();
                    //然后比较和滑动摩擦力的大小
                    F_tcpthis=min(abs(F_tls),abs(miu*F_ncpthis))*UnitTangentVector;
                    F_tcp=F_tcp+F_tcpthis;
                    //然后更新相对位移；
                    if(abs(F_tls)<abs(miu*F_ncpthis)) RelativeTanDistance[i][j]=newRelativeDistance;
                    else RelativeTanDistance[i][j]=-(abs(miu*F_ncpthis)*UnitTangentVector+MyACTMs[i][nparticles].d_t*g_tcp)*(1.0/MyACTMs[i][nparticles].k_t);
                    //处理力矩
                    M_cp=M_cp+((radius*UnitNormalVector)^F_tcpthis);
                }  
                else if(distance>0.0&&RelativeTanDistance[i][j]!=VECTOR(0,0,0))
                {
                    RelativeTanDistance[i][j]=VECTOR(0,0,0);
                } 
            }
        }
       //修改i球方程的右端项
       MyParticles[i].force=F_ncp+F_g+F_f+F_IB+F_tcp;
       MyParticles[i].moment=M_cp+M_IB;
    }
}

void collision::ParticleStatusChange()
{
    
    for(llu i=0;i<nparticles;i++)
    {        
        //就更新好了下一时刻的速度和坐标，这样我就可以算力的时候用到的是同一时刻的坐标和速度
        MyParticles[i].a=MyParticles[i].force*(1.0/MyParticles[i].mass);
        MyParticles[i].alpha=MyParticles[i].moment*(1.0/MyParticles[i].inertia);
        nowstep++;
        VECTOR oldX=MyParticles[i].X,oldV=MyParticles[i].V;
        MyParticles[i].V=MyParticles[i].V+MyParticles[i].a*delta_t;
        MyParticles[i].X=MyParticles[i].X+MyParticles[i].V*delta_t; 
       if(oldX.z>0.000125 && MyParticles[i].X.z<0.000125)
        {oldV=oldV+(MyParticles[i].omega^(VECTOR(0,0,-1)*MyParticles[i].radius));ss<<oldV.x/-oldV.z<<" ";}
        MyParticles[i].omega=MyParticles[i].omega+MyParticles[i].alpha*delta_t;
    }
}

void collision::output()
{
    //先只输出纵坐标
    // ss<<nowstep*delta_t<<" ";
    // for(llu i=0;i<nparticles;i++)
    // ss<<fixed<<setprecision(10)<<MyParticles[i].X.z<<" "<<MyParticles[i].V;
    // ss<<endl;
}

void BoundaryBox::setBoundary(bool leftExists, double leftCoord,  
                     bool rightExists, double rightCoord,  
                     bool bottomExists, double bottomCoord,  
                     bool topExists, double topCoord,  
                     bool frontExists, double frontCoord,  
                     bool backExists, double backCoord)
{
    left = Boundary(leftExists, leftCoord);  
    right = Boundary(rightExists, rightCoord);  
    bottom = Boundary(bottomExists, bottomCoord);  
    top = Boundary(topExists, topCoord);  
    front = Boundary(frontExists, frontCoord);  
    back = Boundary(backExists, backCoord);    
}

VECTOR CalculateRelativeV(const Particle &a, const Particle &b)
{
   // 计算两个球心之间的单位向量 n  
    VECTOR n = (b.X - a.X).Unit(); // 假设 X 表示位置  
  
    // 计算两个小球之间的相对速度 g  
    VECTOR g = a.V - b.V;  
  
    // 计算 R_a * (omega_a ^ n) 和 R_b * (omega_b ^ n)  
    VECTOR temp1 = a.omega ^ n;  
    VECTOR temp2 = b.omega ^ n;  
    VECTOR result1 = a.radius * temp1;  
    VECTOR result2 = b.radius * temp2;  
  
    // 计算最终结果  
    VECTOR ans = g + result1 + result2;  
  
    return ans;     
}
