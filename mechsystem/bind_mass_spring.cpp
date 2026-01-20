#include <sstream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "mass_spring.hpp"
#include "Newmark.hpp"

namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(std::vector<Mass<3>>);
PYBIND11_MAKE_OPAQUE(std::vector<Fix<3>>);
PYBIND11_MAKE_OPAQUE(std::vector<Spring>);
PYBIND11_MAKE_OPAQUE(std::vector<DistanceConstraint<3>>);

PYBIND11_MODULE(mass_spring, m) {
    m.doc() = "mass-spring-system simulator"; 

    py::class_<Mass<2>> (m, "Mass2d")
          .def_property("mass",
                    [](Mass<2> & m) { return m.mass; },
                    [](Mass<2> & m, double mass) { m.mass = mass; })
      .def_property_readonly("pos",
                             [](Mass<2> & m) { return m.pos.data(); });

      ;
      
    m.def("Mass", [](double m, std::array<double,2> p)
    {
      return Mass<2>{m, { p[0], p[1] }};
    });

    
    py::class_<Mass<3>> (m, "Mass3d")
      .def_property("mass",
                    [](Mass<3> & m) { return m.mass; },
                    [](Mass<3> & m, double mass) { m.mass = mass; })
      .def_property_readonly("pos",
                             [](Mass<3> & m) { return m.pos.data(); })
      .def_property("vel",
                    [](Mass<3> & m) { return m.vel.data(); },
                    [](Mass<3> & m, std::array<double,3> v) { m.vel = Vec<3>{v[0], v[1], v[2]}; })
      .def_property("acc",
                    [](Mass<3> & m) { return m.acc.data(); },
                    [](Mass<3> & m, std::array<double,3> a) { m.acc = Vec<3>{a[0], a[1], a[2]}; });
    ;

    m.def("Mass", [](double m, std::array<double,3> p)
    {
      return Mass<3>{m, { p[0], p[1], p[2] }};
    });



    py::class_<Fix<2>> (m, "Fix2d")
      .def_property_readonly("pos",
                             [](Fix<2> & f) { return f.pos.data(); });

    m.def("Fix", [](std::array<double,2> p)
    {
      return Fix<2>{ { p[0], p[1] } };
    });


    
    py::class_<Fix<3>> (m, "Fix3d")
      .def_property_readonly("pos",
                             [](Fix<3> & f) { return f.pos.data(); });
    
    m.def("Fix", [](std::array<double,3> p)
    {
      return Fix<3>{ { p[0], p[1], p[2] } };
    });

    py::class_<Connector> (m, "Connector");

    py::class_<Spring> (m, "Spring")
      .def(py::init<double, double, std::array<Connector,2>>())
      .def_property_readonly("connectors",
                             [](Spring & s) { return s.connectors; })
      ;

    py::class_<DistanceConstraint<3>> (m, "DistanceConstraint")
      .def(py::init<Connector, Connector, double>())
      .def_readwrite("lambda", &DistanceConstraint<3>::lambda)
      .def_readwrite("target_distance", &DistanceConstraint<3>::target_distance)
      .def_property_readonly("connectors",
                             [](DistanceConstraint<3> & c) { return c.connectors; })
      ;

    
    py::bind_vector<std::vector<Mass<3>>>(m, "Masses3d");
    py::bind_vector<std::vector<Fix<3>>>(m, "Fixes3d");
    py::bind_vector<std::vector<Spring>>(m, "Springs");
    py::bind_vector<std::vector<DistanceConstraint<3>>>(m, "DistanceConstraints");        
    
    
    py::class_<MassSpringSystem<2>> (m, "MassSpringSystem2d")
      .def(py::init<>())
      .def("add", [](MassSpringSystem<2> & mss, Mass<2> m) { return mss.addMass(m); })
      ;
      
        
    py::class_<MassSpringSystem<3>> (m, "MassSpringSystem3d")
      .def(py::init<>())
      .def("__str__", [](MassSpringSystem<3> & mss) {
        std::stringstream sstr;
        sstr << mss;
        return sstr.str();
      })
      .def_property("gravity", [](MassSpringSystem<3> & mss) { return mss.getGravity(); },
                    [](MassSpringSystem<3> & mss, std::array<double,3> g) { mss.setGravity(Vec<3>{g[0],g[1],g[2]}); })
      .def("add", [](MassSpringSystem<3> & mss, Mass<3> m) { return mss.addMass(m); })
      .def("add", [](MassSpringSystem<3> & mss, Fix<3> f) { return mss.addFix(f); })
      .def("add", [](MassSpringSystem<3> & mss, Spring s) { return mss.addSpring(s); })
      .def("add_constraint", [](MassSpringSystem<3> & mss, DistanceConstraint<3> c) { return mss.addConstraint(c); })
      .def_property_readonly("masses", [](MassSpringSystem<3> & mss) -> auto& { return mss.masses(); })
      .def_property_readonly("fixes", [](MassSpringSystem<3> & mss) -> auto& { return mss.fixes(); })
      .def_property_readonly("springs", [](MassSpringSystem<3> & mss) -> auto& { return mss.springs(); })
      .def_property_readonly("constraints", [](MassSpringSystem<3> & mss) -> auto& { return mss.constraints(); })
      .def("__getitem__", [](MassSpringSystem<3> mss, Connector & c) {
        if (c.type==Connector::FIX) return py::cast(mss.fixes()[c.nr]);
        else return py::cast(mss.masses()[c.nr]);
      })
      
      .def("getState", [] (MassSpringSystem<3> & mss) {
        size_t npos = 3 * mss.masses().size();
        size_t nconstraints = mss.constraints().size();
        size_t ntotal = npos + nconstraints;
        
        Vector<> x(ntotal);
        Vector<> dx(ntotal);
        Vector<> ddx(ntotal);
        mss.getState (x, dx, ddx);
        return std::vector<double>(x);
      })

      .def("set_mass_velocity", [](MassSpringSystem<3> & mss, size_t idx, std::array<double,3> v) {
        if (idx < mss.masses().size()) {
          mss.masses()[idx].vel = Vec<3>{v[0], v[1], v[2]};
        }
      })
      .def("get_mass_velocity", [](MassSpringSystem<3> & mss, size_t idx) {
        if (idx < mss.masses().size()) {
          return mss.masses()[idx].vel.data();
        }
        return std::array<double,3>{0, 0, 0};
      })
      .def("simulate", [](MassSpringSystem<3> & mss, double tend, size_t steps) {
        size_t npos = 3 * mss.masses().size();
        size_t nconstraints = mss.constraints().size();
        size_t ntotal = npos + nconstraints;
        
        Vector<> x(ntotal);
        Vector<> dx(ntotal);
        Vector<> ddx(ntotal);
        mss.getState (x, dx, ddx);

        auto mss_func = std::make_shared<MSS_Function<3>> (mss);
        auto mass = std::make_shared<MSS_MassMatrix<3>> (mss);

        SolveODE_Alpha(tend, steps, 0.8, x, dx, ddx, mss_func, mass);

        mss.setState (x, dx, ddx);  
    });


  
    
}
