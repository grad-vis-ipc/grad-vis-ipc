// #include <time.h> /* time_t, struct tm, difftime, time, mktime */

#include <chrono>
#include <fstream>
#include <iostream>
#include <thread>

#include "icecream.hpp"
#include "xtensor/xadapt.hpp"

using d_vec = xt::xarray<double>;
using time_pt = std::chrono::_V2::system_clock::time_point;

inline double millisec_diff(time_pt a, time_pt b) {
  return std::chrono::duration_cast<std::chrono::milliseconds>(a - b).count() /
         1000.0;
}

struct SetDispatcher {
  auto to_vec(d_vec a0) { return std::vector<double>(a0.begin(), a0.end()); }
  void step(d_vec a0, double z) { a0 = xt::cos(xt::ones_like(a0) * z); }
};

int main(const int argc, const char* argv[]) {
  icecream::ic.prefix("").line_wrap_width(80);
  auto d1 = SetDispatcher();
  auto time1 = std::chrono::high_resolution_clock::now();

  // auto a0 = xt::ones<double>({10, 10});
  auto x0 = xt::linspace<double>(1.0, 10.0, 5);
  auto y0 = x0;
  auto a0 = xt::meshgrid(x0, y0);

  IC(a0);
  // auto v = std::vector<double>(a0.begin(), a0.end());
  // IC(v);
  // while (true) {
  //   auto time2 = std::chrono::high_resolution_clock::now();
  //   std::this_thread::sleep_for(std::chrono::milliseconds(15));
  //   // d1.step(a0, 1);
  // }

  return EXIT_SUCCESS;
}
