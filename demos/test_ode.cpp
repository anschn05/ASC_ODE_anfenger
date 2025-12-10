
#include <iostream>
#include <fstream> 

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <nonlinfunc.hpp>
#include <timestepper.hpp>
#include <implicitRK.hpp>

using namespace ASC_ode;


class MassSpring : public NonlinearFunction
{
private:
  double mass;
  double stiffness;

public:
  MassSpring(double m, double k) : mass(m), stiffness(k) {}

  size_t dimX() const override { return 2; }
  size_t dimF() const override { return 2; }
  
  void evaluate (VectorView<double> x, VectorView<double> f) const override
  {
    f(0) = x(1);
    f(1) = -stiffness/mass*x(0);
  }
  
  void evaluateDeriv (VectorView<double> x, MatrixView<double> df) const override
  {
    df = 0.0;
    df(0,1) = 1;
    df(1,0) = -stiffness/mass;
  }
};

class RCSpring : public NonlinearFunction
{
private:
  double R;
  double C;

public:
  RCSpring(double m, double k) : R(m), C(k) {}

  size_t dimX() const override { return 2; }
  size_t dimF() const override { return 2; }
  
  void evaluate (VectorView<double> x, VectorView<double> f) const override
  {
    
    f(0) = 1/(R*C)*(cos((M_PI*x(1)*100)) - x(0));
    f(1) = 1;
  }
  
  void evaluateDeriv (VectorView<double> x, MatrixView<double> df) const override
  {
    df = 0.0;
    df(0,0) = -1/(R*C);
    df(0,1) = -1/(R*C)*100*M_PI*sin(M_PI*x(1)*100);
    df(1,1) = 0;
  }
};


int main()
{
  double tend = 0.1;
  int steps = 1000;
  double tau = tend/steps;

  Vector<> y = { 0, 0 };  // initializer list
  auto rhs = std::make_shared<RCSpring>(100.0, 1e-6);
  



/*
  Vector<> Radau(3), RadauWeight(3);
  GaussRadau (Radau, RadauWeight);
  // not sure about weights, comput them via ComputeABfromC
  cout << "Radau = " << Radau << ", weight = " << RadauWeight <<  endl;
*/
 

  // ExplicitEuler stepper(rhs);
  // ImplicitEuler stepper(rhs);

  ImplicitRungeKutta stepper(rhs, Gauss2a, Gauss2b, Gauss2c);

  // Gauss3c .. points tabulated, compute a,b:
  // auto [Gauss3a,Gauss3b] = computeABfromC (Gauss3c);
  // ImplicitRungeKutta stepper(rhs, Gauss3a, Gauss3b, Gauss3c);


  /*
  // arbitrary order Gauss-Legendre
  int stages = 5;
  Vector<> c(stages), b1(stages);
  GaussLegendre(c, b1);

  auto [a, b] = computeABfromC(c);
  ImplicitRungeKutta stepper(rhs, a, b, c);
  */

  /*
  // arbitrary order Radau
  int stages = 5;
  Vector<> c(stages), b1(stages);
  GaussRadau(c, b1);

  auto [a, b] = computeABfromC(c);
  ImplicitRungeKutta stepper(rhs, a, b, c);
  */


  std::ofstream outfile ("output_test_ode.txt");
  std::cout << 0.0 << "  " << y(0) << " " << y(1) << std::endl;
  outfile << 0.0 << "  " << y(0) << " " << y(1) << std::endl;

  for (int i = 0; i < steps; i++)
  {
     stepper.doStep(tau, y);

     std::cout << (i+1) * tau << "  " << y(0) << " " << y(1) << std::endl;
     outfile << (i+1) * tau << "  " << y(0) << " " << y(1) << std::endl;
  }
}
