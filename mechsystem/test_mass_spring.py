import os, sys
here = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(here, "..", "build", "mechsystem")))

from mass_spring import *


mss = MassSpringSystem3d()
mss.gravity = (0,0,-9.81)

mA = mss.add (Mass(1, (0,0,1)))
mB = mss.add (Mass(2, (0.2,0,2)))
f1 = mss.add (Fix( (0,0,0)) )
mss.add (Spring(1, 10, (f1, mA)))
mss.add (Spring(1, 20, (mA, mB)))
A = mss.masses[0].pos
B = mss.masses[1].pos
L0 = ((B[0]-A[0])**2 + (B[1]-A[1])**2 + (B[2]-A[2])**2)**0.5   
# Add distance constraint with Lagrange multipliers
constraint_AB = DistanceConstraint(mA, mB, L0)
mss.add_constraint(constraint_AB)


print ("state = ", mss.getState())

mss.simulate (0.01, 100)

print ("state = ", mss.getState())


mss.simulate (0.01, 100)

print ("state = ", mss.getState())

for m in mss.masses:
    print (m.mass, m.pos)

mss.masses[0].mass = 5

for m in mss.masses:
    print (m.mass, m.pos)

dt = 0.001
nsteps = 20000

with open("output_mass_spring_py.txt", "w") as f:
    for k in range(nsteps+1):
        t = k*dt

        A = mss.masses[0]
        # falls vel verfügbar:
        vz = A.vel[2] if hasattr(A, "vel") else 0.0

        f.write(f"{t} {A.pos[2]} {vz}\n")

        if k < nsteps:
            mss.simulate(dt, 1)
print("Simulation complete, data written to output_mass_spring_py.txt")