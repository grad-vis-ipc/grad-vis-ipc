#pragma once
#include "logistic_regression.hpp"
#include "xtensor/xadapt.hpp"
auto pearson_correlation(const d_vec& x, const d_vec& y) {
  auto x_norm = x - xt::mean(x);
  auto y_norm = y - xt::mean(y);
  auto r_top = xt::sum(x_norm * y_norm);
  auto r_bottom_x = xt::sqrt(xt::sum(xt::pow(x_norm, 2)));
  auto r_bottom_y = xt::sqrt(xt::sum(xt::pow(y_norm, 2)));
  auto r_correlation = r_top / (r_bottom_x * r_bottom_y);
  IC(r_correlation);
  return xt::eval(r_correlation);
}