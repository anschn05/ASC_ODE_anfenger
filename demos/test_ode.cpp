
#include <iostream>
#include <fstream> 

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <nonlinfunc.hpp>
#include <timestepper.hpp>


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
  
  ImplicitEuler stepper(rhs);
  // ImplicitEuler stepper(rhs);

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
