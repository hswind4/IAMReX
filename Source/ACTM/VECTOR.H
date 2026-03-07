#include<cmath>
#include<cstdio>
#include<iostream>
#ifndef VECTOR_H
#define VECTOR_H
class VECTOR{
public:
    double x=0.0,y=0.0,z=0.0;
    VECTOR() = default;
    VECTOR(double,double,double);
    VECTOR Unit() const;
    VECTOR operator+(const VECTOR& V) const;
    VECTOR& operator=(const VECTOR& V);
    VECTOR operator-(const VECTOR& V) const;
    VECTOR operator-() const;
    VECTOR operator*(double scalar) const;
    double operator*(const VECTOR& V) const;
    VECTOR operator^(const VECTOR& V) const; //叉乘
    bool operator==(const VECTOR& V) const;
    bool operator!=(const VECTOR& V) const;
};
double abs(const VECTOR& vec);
VECTOR operator*(double scalar, const VECTOR& vec);
std::ostream& operator<<(std::ostream& os, const VECTOR& vec);
#endif