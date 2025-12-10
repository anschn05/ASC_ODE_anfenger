#ifndef NONLINFUNC_H
#define NONLINFUNC_H

#include <cstddef>
#include <memory>

#include <vector.hpp>
#include <matrix.hpp>
#include "autodiff.hpp"

namespace ASC_ode
{
  using namespace nanoblas;

  class NonlinearFunction
  {
  public:
    virtual ~NonlinearFunction() = default;
    virtual size_t dimX() const = 0;
    virtual size_t dimF() const = 0;
    virtual void evaluate (VectorView<double> x, VectorView<double> f) const = 0;
    virtual void evaluateDeriv (VectorView<double> x, MatrixView<double> df) const = 0;
  };


  class IdentityFunction : public NonlinearFunction
  {
    size_t m_n;
  public:
    IdentityFunction (size_t n) : m_n(n) { } 
    size_t dimX() const override { return m_n; }
    size_t dimF() const override { return m_n; }
    void evaluate (VectorView<double> x, VectorView<double> f) const override
    {
      f = x;
    }

    void evaluateDeriv (VectorView<double> x, MatrixView<double> df) const override
    {
      df = 0.0;
      df.diag() = 1.0;
    }
  };



  class ConstantFunction : public NonlinearFunction
  {
    Vector<> m_val;
  public:
    ConstantFunction(size_t n) : m_val(n) { }
    ConstantFunction(VectorView<double> val) : m_val(val) { }
    void set(VectorView<double> val) { m_val = val; }
    VectorView<double> get() const { return m_val; }
    size_t dimX() const override { return m_val.size(); }
    size_t dimF() const override { return m_val.size(); }
    void evaluate (VectorView<double> x, VectorView<double> f) const override
    {
      f = m_val;
    }
    void evaluateDeriv (VectorView<double> x, MatrixView<double> df) const override
    {
      df = 0.0;
    }
  };

  
  
  class SumFunction : public NonlinearFunction
  {
    std::shared_ptr<NonlinearFunction> m_fa, m_fb;
    double m_faca, m_facb;
  public:
    SumFunction (std::shared_ptr<NonlinearFunction> fa,
                 std::shared_ptr<NonlinearFunction> fb,
                 double faca, double facb)
      : m_fa(fa), m_fb(fb), m_faca(faca), m_facb(facb) { }

    size_t dimX() const override { return m_fa->dimX(); }
    size_t dimF() const override { return m_fa->dimF(); }
    void evaluate (VectorView<double> x, VectorView<double> f) const override
    {
      m_fa->evaluate(x, f);
      f *= m_faca;
      Vector<> tmp(dimF());
      m_fb->evaluate(x, tmp);
      f += m_facb*tmp;
    }
    void evaluateDeriv (VectorView<double> x, MatrixView<double> df) const override
    {
      m_fa->evaluateDeriv(x, df);
      df *= m_faca;
      Matrix<double> tmp(dimF(), dimX());
      m_fb->evaluateDeriv(x, tmp);
      df += m_facb*tmp;
    }
  };


  inline auto operator- (std::shared_ptr<NonlinearFunction> fa, std::shared_ptr<NonlinearFunction> fb)
  {
    return std::make_shared<SumFunction>(fa, fb, 1, -1);
  }

  inline auto operator+ (std::shared_ptr<NonlinearFunction> fa, std::shared_ptr<NonlinearFunction> fb)
  {
    return std::make_shared<SumFunction>(fa, fb, 1, 1);
  }

  class Parameter 
  {
    double m_value;
  public:
    Parameter(double value) : m_value(value) {}
    double get() const { return m_value; }
    void set(double value) { m_value = value; }
  };

  class ScaleFunction : public NonlinearFunction
  {
    std::shared_ptr<NonlinearFunction> m_fa;
    std::shared_ptr<Parameter> m_fac;
  public:
    ScaleFunction (std::shared_ptr<NonlinearFunction> fa,
                   std::shared_ptr<Parameter> fac)
      : m_fa(fa), m_fac(fac) { }

    size_t dimX() const override { return m_fa->dimX(); }
    size_t dimF() const override { return m_fa->dimF(); }
    void evaluate (VectorView<double> x, VectorView<double> f) const override
    {
      m_fa->evaluate(x, f);
      f *= m_fac->get();
   }

    void evaluateDeriv (VectorView<double> x, MatrixView<double> df) const override
    {
      m_fa->evaluateDeriv(x, df);
      df *= m_fac->get();
    }
  };

  inline auto operator* (std::shared_ptr<Parameter> parama, 
                         std::shared_ptr<NonlinearFunction> f)
  {
    return std::make_shared<ScaleFunction>(f, parama);
  }

  inline auto operator* (double a, std::shared_ptr<NonlinearFunction> f)
  {
    return std::make_shared<Parameter>(a) * f;
  } 




  // fa(fb)
  class ComposeFunction : public NonlinearFunction
  {
    std::shared_ptr<NonlinearFunction> m_fa, m_fb;
  public:
    ComposeFunction (std::shared_ptr<NonlinearFunction> fa,
                     std::shared_ptr<NonlinearFunction> fb)
      : m_fa(fa), m_fb(fb) { }

    size_t dimX() const override { return m_fb->dimX(); }
    size_t dimF() const override { return m_fa->dimF(); }
    void evaluate (VectorView<double> x, VectorView<double> f) const override
    {
      Vector<> tmp(m_fb->dimF());
      m_fb->evaluate (x, tmp);
      m_fa->evaluate (tmp, f);
    }
    void evaluateDeriv (VectorView<double> x, MatrixView<double> df) const override
    {
      Vector<> tmp(m_fb->dimF());
      m_fb->evaluate (x, tmp);

      Matrix<double> jaca(m_fa->dimF(), m_fa->dimX());
      Matrix<double> jacb(m_fb->dimF(), m_fb->dimX());

      m_fb->evaluateDeriv(x, jacb);
      m_fa->evaluateDeriv(tmp, jaca);

      df = jaca*jacb;
    }
  };
  
  
  inline auto Compose (std::shared_ptr<NonlinearFunction> fa, std::shared_ptr<NonlinearFunction> fb)
  {
    return make_shared<ComposeFunction> (fa, fb);
  }
  
  class EmbedFunction : public NonlinearFunction
  {
    std::shared_ptr<NonlinearFunction> m_fa;
    size_t m_firstx, m_dimx, m_firstf, m_dimf;
    size_t m_nextx, m_nextf;
  public:
    EmbedFunction (std::shared_ptr<NonlinearFunction> fa,
                   size_t firstx, size_t dimx,
                   size_t firstf, size_t dimf)
      : m_fa(fa),
        m_firstx(firstx), m_dimx(dimx), m_firstf(firstf), m_dimf(dimf),
        m_nextx(m_firstx+m_fa->dimX()), m_nextf(m_firstf+m_fa->dimF())
    { }

    size_t dimX() const override { return m_dimx; }
    size_t dimF() const override { return m_dimf; }
    void evaluate (VectorView<double> x, VectorView<double> f) const override
    {
      f = 0.0;
      m_fa->evaluate(x.range(m_firstx, m_nextx), f.range(m_firstf, m_nextf));
    }
    void evaluateDeriv (VectorView<double> x, MatrixView<double> df) const override
    {
      df = 0;
      m_fa->evaluateDeriv(x.range(m_firstx, m_nextx),
                        df.rows(m_firstf, m_nextf).cols(m_firstx, m_nextx));
    }
  };

  
  class Projector : public NonlinearFunction
  {
    size_t m_size, m_first, m_next;
  public:
    Projector (size_t size, 
               size_t first, size_t next)
      : m_size(size), m_first(first), m_next(next) { }

    size_t dimX() const override { return m_size; }
    size_t dimF() const override { return m_size; }
    void evaluate (VectorView<double> x, VectorView<double> f) const override
    {
      f = 0.0;
      f.range(m_first, m_next) = x.range(m_first, m_next);
    }
    void evaluateDeriv (VectorView<double> x, MatrixView<double> df) const override
    {
      df = 0.0;
      df.diag().range(m_first, m_next) = 1;
    }
  };


  template <typename NLF>
  class NonlinearFunctionAutoDif : public NonlinearFunction {
  public:

   void evaluate(VectorView<double> x, VectorView<double> f) const override {
    static_cast<const NLF*>(this) -> T_evaluate(x, f);
  }

  void evaluateDeriv (VectorView<double> x, MatrixView<double> df) const override {
    Vector<AutoDiff<1>> adx(dimX());
    Vector<AutoDiff<1>> adf(dimF());

    for (int i = 0; i < dimX(); i++) {
      for (int j = 0; j < dimX(); j++)
        adx(j) = x(j);
      adx(i) = Variable<0>(x(i));
      static_cast<const NLF*>(this) -> T_evaluate(adx, adf);
      for (int j = 0; j < dimF(); j++)
        df(j,i) = adf(j).deriv()[0];
    }
  }  
};

  // Beispiel-Klasse für NonlinearFunctionAutoDif
  class MyFunc : public NonlinearFunctionAutoDif<MyFunc> {
  public:
    size_t dimX() const override { return 2; }
    size_t dimF() const override { return 2; }

    template <typename T>
    void T_evaluate(VectorView<T> x, VectorView<T> f) const {
      f(0) = x(0) * x(0) + x(1);
      f(1) = x(0) - x(1) * x(1);
    }
  };

  // Pendel-Funktion mit Autodiff
  class PendulumAD : public NonlinearFunctionAutoDif<PendulumAD> {
  private:
    double m_length;
    double m_gravity;

  public:
    PendulumAD(double length, double gravity = 9.81) 
      : m_length(length), m_gravity(gravity) {}

    size_t dimX() const override { return 2; }
    size_t dimF() const override { return 2; }

    template <typename T>
    void T_evaluate(VectorView<T> x, VectorView<T> f) const {
      f(0) = x(1);
      f(1) = T(m_gravity / m_length) * T(-1.0) * sin(x(0));
    }
  };

  
}

#endif