#include <time.h> /* time_t, struct tm, difftime, time, mktime */

#include <chrono>
#include <fstream>
#include <iostream>
#include <thread>

#include "icecream.hpp"
#include "xtensor/xadapt.hpp"

using d_vec = xt::xarray<double>;

struct Dispatcher {
  d_vec a0;
  Dispatcher(double a = 1, double b = 1, double c = 1) { a0 = {a, b, c}; }
  auto to_vec() { return std::vector<double>(a0.begin(), a0.end()); }
  void step(double z) { a0 = xt::cos(xt::ones_like(a0) * z); }
};

int main(const int argc, const char* argv[]) {
  auto d1 = Dispatcher();
  // std::ios_base::sync_with_stdio(false);
  // std::cin.tie(NULL);

  auto time1 = std::chrono::high_resolution_clock::now();

  while (true) {
    std::cout << d1.a0[0] << ',' << d1.a0[1] << ',' << d1.a0[2] << ','
              << std::endl;
    auto time2 = std::chrono::high_resolution_clock::now();
    double s =
        std::chrono::duration_cast<std::chrono::milliseconds>(time2 - time1)
            .count() /
        1000.0;
    std::this_thread::sleep_for(std::chrono::milliseconds(15));

    d1.step(s);
  }

  return EXIT_SUCCESS;
}
