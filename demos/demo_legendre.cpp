#include <iostream>
#include <autodiff.hpp>
#include <vector>


using namespace ASC_ode;





template <typename T>
void LegendrePolynomials(int n, T x, std::vector<T>& P) {
    if (n < 0) {
        P.clear();
        return;
    }
    P.resize(n + 1);
    P[0] = T(1);
    if (n == 0) return;
    P[1] = x;
    for (int k = 2; k <= n; ++k) {
        P[k] = ((T(2 * k - 1) * x * P[k - 1]) - T(k - 1) * P[k - 2]) / T(k);
    }
}

int main(){
    double x = 1.0;
    AutoDiff<2> adx = Variable<0>(x);

    using AD1 = AutoDiff<1>;
    std::vector<AD1> P;
    LegendrePolynomials(10, AD1(0.7), P);

    for (size_t i = 0; i < P.size(); ++i)
        std::cout << "P[" << i << "] = " << P[i] << "\n";

    

    return 0;
}
