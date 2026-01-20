#ifndef MASS_SPRING_HPP
#define MASS_SPRING_HPP

#include <nonlinfunc.hpp>
#include <timestepper.hpp>

using namespace ASC_ode;

#include <vector.hpp>
using namespace nanoblas;


template <int D>
class Mass
{
public:
  double mass;
  Vec<D> pos;
  Vec<D> vel = 0.0;
  Vec<D> acc = 0.0;
};


template <int D>
class Fix
{
public:
  Vec<D> pos;
};


class Connector
{
public:
  enum CONTYPE { FIX=1, MASS=2 };
  CONTYPE type;
  size_t nr;
};

std::ostream & operator<< (std::ostream & ost, const Connector & con)
{
  ost << "type = " << int(con.type) << ", nr = " << con.nr;
  return ost;
}

class Spring
{
public:
  double length;  
  double stiffness;
  std::array<Connector,2> connectors;
};

template <int D>
class DistanceConstraint
{
public:
  std::array<Connector,2> connectors;
  double target_distance;
  double lambda = 0.0;  // Lagrange multiplier
  
  DistanceConstraint(Connector c1, Connector c2, double dist)
    : connectors{c1, c2}, target_distance(dist) {}
};

template <int D>
class MassSpringSystem
{
  std::vector<Fix<D>> m_fixes;
  std::vector<Mass<D>> m_masses;
  std::vector<Spring> m_springs;
  std::vector<DistanceConstraint<D>> m_constraints;
  Vec<D> m_gravity=0.0;
public:
  void setGravity (Vec<D> gravity) { m_gravity = gravity; }
  Vec<D> getGravity() const { return m_gravity; }

  Connector addFix (Fix<D> p)
  {
    m_fixes.push_back(p);
    return { Connector::FIX, m_fixes.size()-1 };
  }

  Connector addMass (Mass<D> m)
  {
    m_masses.push_back (m);
    return { Connector::MASS, m_masses.size()-1 };
  }
  
  size_t addSpring (Spring s) 
  {
    m_springs.push_back (s); 
    return m_springs.size()-1;
  }

  size_t addConstraint (DistanceConstraint<D> c)
  {
    m_constraints.push_back (c);
    return m_constraints.size()-1;
  }

  auto & fixes() { return m_fixes; } 
  auto & masses() { return m_masses; } 
  auto & springs() { return m_springs; }
  auto & constraints() { return m_constraints; }

  void getState (VectorView<> values, VectorView<> dvalues, VectorView<> ddvalues)
  {
    size_t npos = D * m_masses.size();
    size_t nconstraints = m_constraints.size();
    
    auto valmat = values.range(0, npos).asMatrix(m_masses.size(), D);
    auto dvalmat = dvalues.range(0, npos).asMatrix(m_masses.size(), D);
    auto ddvalmat = ddvalues.range(0, npos).asMatrix(m_masses.size(), D);

    for (size_t i = 0; i < m_masses.size(); i++)
      {
        valmat.row(i) = m_masses[i].pos;
        dvalmat.row(i) = m_masses[i].vel;
        ddvalmat.row(i) = m_masses[i].acc;
      }
    
    // Store Lagrange multipliers
    for (size_t i = 0; i < nconstraints; i++)
      {
        values(npos + i) = m_constraints[i].lambda;
        dvalues(npos + i) = 0.0;
        ddvalues(npos + i) = 0.0;
      }
  }

  void setState (VectorView<> values, VectorView<> dvalues, VectorView<> ddvalues)
  {
    size_t npos = D * m_masses.size();
    size_t nconstraints = m_constraints.size();
    
    auto valmat = values.range(0, npos).asMatrix(m_masses.size(), D);
    auto dvalmat = dvalues.range(0, npos).asMatrix(m_masses.size(), D);
    auto ddvalmat = ddvalues.range(0, npos).asMatrix(m_masses.size(), D);

    for (size_t i = 0; i < m_masses.size(); i++)
      {
        m_masses[i].pos = valmat.row(i);
        m_masses[i].vel = dvalmat.row(i);
        m_masses[i].acc = ddvalmat.row(i);
      }
    
    // Retrieve Lagrange multipliers
    for (size_t i = 0; i < nconstraints; i++)
      {
        m_constraints[i].lambda = values(npos + i);
      }
  }
};

template <int D>
std::ostream & operator<< (std::ostream & ost, MassSpringSystem<D> & mss)
{
  ost << "fixes:" << std::endl;
  for (auto f : mss.fixes())
    ost << f.pos << std::endl;

  ost << "masses: " << std::endl;
  for (auto m : mss.masses())
    ost << "m = " << m.mass << ", pos = " << m.pos << std::endl;

  ost << "springs: " << std::endl;
  for (auto sp : mss.springs())
    ost << "length = " << sp.length << ", stiffness = " << sp.stiffness
        << ", C1 = " << sp.connectors[0] << ", C2 = " << sp.connectors[1] << std::endl;

  ost << "constraints: " << std::endl;
  for (auto c : mss.constraints())
    ost << "target_distance = " << c.target_distance << ", lambda = " << c.lambda
        << ", C1 = " << c.connectors[0] << ", C2 = " << c.connectors[1] << std::endl;
  
  return ost;
}


template <int D>
class MSS_Function : public NonlinearFunction
{
  MassSpringSystem<D> & mss;
public:
  MSS_Function (MassSpringSystem<D> & _mss)
    : mss(_mss) { }

  virtual size_t dimX() const override { return D*mss.masses().size() + mss.constraints().size(); }
  virtual size_t dimF() const override{ return D*mss.masses().size() + mss.constraints().size(); }

  virtual void evaluate (VectorView<double> x, VectorView<double> f) const override
  {
    f = 0.0;

    size_t npos = D * mss.masses().size();
    size_t nconstraints = mss.constraints().size();
    
    auto xmat = x.range(0, npos).asMatrix(mss.masses().size(), D);
    auto fmat = f.range(0, npos).asMatrix(mss.masses().size(), D);
    auto lambda = x.range(npos, npos + nconstraints);

    // Gravitational force (actual force, not acceleration)
    for (size_t i = 0; i < mss.masses().size(); i++)
      fmat.row(i) = mss.masses()[i].mass*mss.getGravity();

    // Spring forces (actual forces, not accelerations)
    for (auto spring : mss.springs())
      {
        auto [c1,c2] = spring.connectors;
        Vec<D> p1, p2;
        if (c1.type == Connector::FIX)
          p1 = mss.fixes()[c1.nr].pos;
        else
          p1 = xmat.row(c1.nr);
        if (c2.type == Connector::FIX)
          p2 = mss.fixes()[c2.nr].pos;
        else
          p2 = xmat.row(c2.nr);

        double force = spring.stiffness * (norm(p1-p2)-spring.length);
        Vec<D> dir12 = 1.0/norm(p1-p2) * (p2-p1);
        if (c1.type == Connector::MASS)
          fmat.row(c1.nr) += force*dir12;
        if (c2.type == Connector::MASS)
          fmat.row(c2.nr) -= force*dir12;
      }

    // Constraint forces: -λ·∇g (from Lagrangian: m·ẍ = -∇U - λ·∇g)
    // For constraint g(x) = |x1 - x2|² - l², we have:
    // ∇g/∇x1 = 2(x1 - x2)
    // ∇g/∇x2 = 2(x2 - x1) = -∇g/∇x1
    for (size_t ic = 0; ic < nconstraints; ic++)
      {
        auto & constraint = mss.constraints()[ic];
        auto [c1, c2] = constraint.connectors;
        
        Vec<D> p1, p2;
        if (c1.type == Connector::FIX)
          p1 = mss.fixes()[c1.nr].pos;
        else
          p1 = xmat.row(c1.nr);
        if (c2.type == Connector::FIX)
          p2 = mss.fixes()[c2.nr].pos;
        else
          p2 = xmat.row(c2.nr);

        Vec<D> grad_g = 2.0 * (p1 - p2);
        
        if (c1.type == Connector::MASS)
          fmat.row(c1.nr) -= lambda(ic) * grad_g;
        if (c2.type == Connector::MASS)
          fmat.row(c2.nr) += lambda(ic) * grad_g;
      }

    // Divide by mass to get accelerations (ẍ = F/m)
    for (size_t i = 0; i < mss.masses().size(); i++)
      fmat.row(i) *= 1.0/mss.masses()[i].mass;

    // Constraint equations: g(x) = |x1 - x2|² - l² = 0
    for (size_t ic = 0; ic < nconstraints; ic++)
      {
        auto & constraint = mss.constraints()[ic];
        auto [c1, c2] = constraint.connectors;
        
        Vec<D> p1, p2;
        if (c1.type == Connector::FIX)
          p1 = mss.fixes()[c1.nr].pos;
        else
          p1 = xmat.row(c1.nr);
        if (c2.type == Connector::FIX)
          p2 = mss.fixes()[c2.nr].pos;
        else
          p2 = xmat.row(c2.nr);

        double dist_sq = dot(p1 - p2, p1 - p2);
        double target_sq = constraint.target_distance * constraint.target_distance;
        f(npos + ic) = dist_sq - target_sq;
      }
  }
  
  virtual void evaluateDeriv (VectorView<double> x, MatrixView<double> df) const override
  {
    // TODO: exact differentiation
    double eps = 1e-8;
    Vector<> xl(dimX()), xr(dimX()), fl(dimF()), fr(dimF());
    for (size_t i = 0; i < dimX(); i++)
      {
        xl = x;
        xl(i) -= eps;
        xr = x;
        xr(i) += eps;
        evaluate (xl, fl);
        evaluate (xr, fr);
        df.col(i) = 1/(2*eps) * (fr-fl);
      }
  }
  
};

// Mass matrix for DAE system with constraints
// Has form: [M  0]
//           [0  0]
// where M is the diagonal mass matrix and 0 blocks correspond to Lagrange multipliers
template <int D>
class MSS_MassMatrix : public NonlinearFunction
{
  MassSpringSystem<D> & mss;
public:
  MSS_MassMatrix (MassSpringSystem<D> & _mss)
    : mss(_mss) { }

  virtual size_t dimX() const override { return D*mss.masses().size() + mss.constraints().size(); }
  virtual size_t dimF() const override { return D*mss.masses().size() + mss.constraints().size(); }

  virtual void evaluate (VectorView<double> x, VectorView<double> f) const override
  {
    size_t npos = D * mss.masses().size();
    size_t nconstraints = mss.constraints().size();
    
    auto xmat = x.range(0, npos).asMatrix(mss.masses().size(), D);
    auto fmat = f.range(0, npos).asMatrix(mss.masses().size(), D);

    // Apply mass matrix to position components
    for (size_t i = 0; i < mss.masses().size(); i++)
      fmat.row(i) = mss.masses()[i].mass * xmat.row(i);

    // Zero for Lagrange multipliers
    for (size_t i = 0; i < nconstraints; i++)
      f(npos + i) = 0.0;
  }
  
  virtual void evaluateDeriv (VectorView<double> x, MatrixView<double> df) const override
  {
    df = 0.0;
    size_t npos = D * mss.masses().size();
    
    // Mass matrix is constant, so derivative is just the matrix itself
    for (size_t i = 0; i < mss.masses().size(); i++)
      for (size_t d = 0; d < D; d++)
        df(i*D + d, i*D + d) = mss.masses()[i].mass;
    
    // Zero blocks for constraint rows and columns (already zero from initialization)
  }
};

#endif