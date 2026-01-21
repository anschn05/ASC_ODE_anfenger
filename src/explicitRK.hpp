#include <memory>
#include <vector.hpp>
#include <matrix.hpp>
#include <nonlinfunc.hpp>

namespace ASC_ode
{
  using namespace nanoblas;

  class ExplicitRungeKutta : public TimeStepper
  {
    Matrix<> m_a;
    Vector<> m_b, m_c;
    std::shared_ptr<NonlinearFunction> m_rhs;

    int m_stages;      // Anzahl der Stufen s
    int m_n;           // Dimension des Zustandsvektors

    Vector<> m_k;      // alle k_j hintereinander: Größe s * n
    Vector<> m_ytemp;  // Zwischenwert y_j

  public:
    ExplicitRungeKutta(std::shared_ptr<NonlinearFunction> rhs,
                       const Matrix<> & a,
                       const Vector<> & b,
                       const Vector<> & c)
      : TimeStepper(rhs),
        m_a(a), m_b(b), m_c(c),
        m_rhs(rhs),
        m_stages(int(c.size())),
        m_n(int(rhs->dimX())),
        m_k(m_stages * m_n),
        m_ytemp(m_n)
    {
      // Optional: ein paar sanity-checks (kannst du auch weglassen)
      // assert(m_a.Rows() == m_a.Cols());
      // assert(int(m_a.Rows()) == m_stages);
      // assert(int(m_b.size()) == m_stages);
      // assert(int(m_c.size()) == m_stages);
      // assert(rhs->dimF() == rhs->dimX());
    }

    void doStep(double tau, VectorView<double> y) override
    {
      // Stufen k_j berechnen
      for (int j = 0; j < m_stages; ++j)
      {
        // ytemp = y_n
        for (int i = 0; i < m_n; ++i)
          m_ytemp(i) = y(i);

        // ytemp += tau * sum_{l<j} a_{j,l} * k_l
        // (nur untere Dreiecksmatrix von A wird genutzt)
        for (int l = 0; l < j; ++l)
        {
          double a_jl = m_a(j, l);
          if (a_jl == 0.0) continue;

          auto k_l = m_k.range(l * m_n, (l + 1) * m_n);
          for (int i = 0; i < m_n; ++i)
            m_ytemp(i) += tau * a_jl * k_l(i);
        }

        // k_j = f( ytemp )
        auto k_j = m_k.range(j * m_n, (j + 1) * m_n);
        m_rhs->evaluate(m_ytemp, k_j);
      }

      // y_{n+1} = y_n + tau * sum_j b_j * k_j
      // m_ytemp als Inkrement verwenden
      for (int i = 0; i < m_n; ++i)
        m_ytemp(i) = 0.0;

      for (int j = 0; j < m_stages; ++j)
      {
        double b_j = m_b(j);
        if (b_j == 0.0) continue;

        auto k_j = m_k.range(j * m_n, (j + 1) * m_n);
        for (int i = 0; i < m_n; ++i)
          m_ytemp(i) += b_j * k_j(i);
      }

      for (int i = 0; i < m_n; ++i)
        y(i) += tau * m_ytemp(i);
    }
  };

} // namespace ASC_ode