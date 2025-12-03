#ifndef TIMERSTEPPER_HPP
#define TIMERSTEPPER_HPP

#include <functional>
#include <exception>

#include "Newton.hpp"


namespace ASC_ode
{
  
  class TimeStepper
  { 
  protected:
    std::shared_ptr<NonlinearFunction> m_rhs;
  public:
    TimeStepper(std::shared_ptr<NonlinearFunction> rhs) : m_rhs(rhs) {}
    virtual ~TimeStepper() = default;
    virtual void doStep(double tau, VectorView<double> y) = 0;
  };

  class ExplicitEuler : public TimeStepper
  {
    Vector<> m_vecf;
  public:
    ExplicitEuler(std::shared_ptr<NonlinearFunction> rhs) 
    : TimeStepper(rhs), m_vecf(rhs->dimF()) {}
    void doStep(double tau, VectorView<double> y) override
    {
      this->m_rhs->evaluate(y, m_vecf);
      y += tau * m_vecf;
    }
  };

  class ImplicitEuler : public TimeStepper
  {
    std::shared_ptr<NonlinearFunction> m_equ;
    std::shared_ptr<Parameter> m_tau;
    std::shared_ptr<ConstantFunction> m_yold;
  public:
    ImplicitEuler(std::shared_ptr<NonlinearFunction> rhs) 
    : TimeStepper(rhs), m_tau(std::make_shared<Parameter>(0.0)) 
    {
      m_yold = std::make_shared<ConstantFunction>(rhs->dimX());
      auto ynew = std::make_shared<IdentityFunction>(rhs->dimX());
      m_equ = ynew - m_yold - m_tau * m_rhs;
    }

    void doStep(double tau, VectorView<double> y) override
    {
      m_yold->set(y);
      m_tau->set(tau);
      NewtonSolver(m_equ, y);
    }
  };

  class ImprovedEuler : public TimeStepper
{
  Vector<> m_vecf, m_vecf2, m_y_temp;
public:
  ImprovedEuler(std::shared_ptr<NonlinearFunction> rhs) 
  : TimeStepper(rhs), m_vecf(rhs->dimF()), m_vecf2(rhs->dimF()), m_y_temp(rhs->dimX()) {}
  
  void doStep(double tau, VectorView<double> y) override
  {
    // Erster Schritt: k1 = f(y_n)
    this->m_rhs->evaluate(y, m_vecf);
    
    // Zweiter Schritt: k2 = f(y_n + tau * k1)
    m_y_temp = y + tau * m_vecf;
    this->m_rhs->evaluate(m_y_temp, m_vecf2);
    
    // Update: y_{n+1} = y_n + tau/2 * (k1 + k2)
    y += (tau / 2.0) * (m_vecf + m_vecf2);
  }
};

  

}


#endif
