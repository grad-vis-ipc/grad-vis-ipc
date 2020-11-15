#pragma once
#include "stats.cpp"

namespace kern {

auto sq_adj_pair(const d_vec& X) {
  auto kernel_train = xt::eval(X);
  auto n_cols = X.shape()[1];
  // append squares: x^2
  for (t i = 0; i < n_cols - 1; ++i) {
    auto squares =
        xt::pow(xt::eval(xt::view(X, xt::all(), xt::range(i, i + 1))), 2);
    kernel_train = xt::hstack(xt::xtuple(kernel_train, squares));
  }
  // append element-wise adjacent pairs: xy
  for (t i = 0; i < n_cols - 2; ++i) {
    auto prod = xt::eval(xt::view(X, xt::all(), xt::range(i, i + 1))) *
                xt::eval(xt::view(X, xt::all(), xt::range(i + 1, i + 2)));
    kernel_train = xt::hstack(xt::xtuple(kernel_train, prod));
  }
  return kernel_train;
}

auto pearson_r(const d_vec& X, const d_vec& y) {
  // raise each row to the power of its correlation
  auto kernel_train = xt::eval(X);
  auto n_cols = X.shape()[1];
  for (t i = 0; i < n_cols; ++i) {
    auto r_val = pearson_correlation(xt::eval(xt::col(X, i)), y);
    auto squares =
        xt::pow(xt::eval(xt::view(X, xt::all(), xt::range(i, i + 1))),
                10 * xt::abs(r_val));
    kernel_train = xt::hstack(xt::xtuple(kernel_train, squares));
  }
  return xt::eval(
      xt::view(kernel_train, xt::all(), xt::range(n_cols, 2 * n_cols)));
}

}  // namespace kern