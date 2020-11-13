#include "logistic_regression.hpp"
#include "xtensor/xadapt.hpp"

using d_vec = xt::xarray<double>;
using b_vec = xt::xarray<bool>;
using t = std::size_t;
using d = double;

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
        xt::pow(xt::eval(xt::view(X, xt::all(), xt::range(i, i + 1))), 1+r_val/2.0);
    kernel_train = xt::hstack(xt::xtuple(kernel_train, squares));
  }
  return xt::eval(
      xt::view(kernel_train, xt::all(), xt::range(n_cols, 2 * n_cols)));
}

}  // namespace kern

int main(const int argc, const char* argv[]) {
  if (argc < 5) {
    std::cout << "Usage: " << argv[0] << " <train> <test> <iterations> <step>"
              << std::endl;
    return EXIT_FAILURE;
  }
  const auto [train_fname, test_fname, N_ITER, LEARNING_RATE] =
      std::make_tuple(argv[1], argv[2], atoi(argv[3]), std::stod(argv[4]));

  // load csv from files
  std::ifstream train(train_fname);  // training set
  auto training_set_raw = xt::load_csv<double>(train);
  train.close();

  std::ifstream test(test_fname);  // test set
  auto test_set_raw = xt::load_csv<double>(test);
  test.close();

  auto train_labels = xt::col(training_set_raw, 0);
  auto train_set =
      xt::view(training_set_raw, xt::all(), xt::range(1, xt::placeholders::_));

  auto test_labels = xt::eval(xt::col(test_set_raw, 0));
  auto test_set =
      xt::view(test_set_raw, xt::all(), xt::range(1, xt::placeholders::_));

  // auto kernel_train = kern::pearson_r(train_set, train_labels);
  auto kernel_train = kern::pearson_r(kern::sq_adj_pair(train_set), train_labels);

  auto kern_weights =
      logistic_regression(kernel_train, train_labels, N_ITER, LEARNING_RATE);

  IC(kernel_train, kernel_train.shape());
  // auto scores = xt::linalg::dot(xt::eval(test_set), kern_weights);
  auto scores = xt::linalg::dot(kern::pearson_r(kern::sq_adj_pair(test_set), test_labels), kern_weights);

  auto logits = 1.0 / (1.0 + xt::exp(-1 * scores));
  d_vec actual = xt::eval(test_labels);
  b_vec predictions = logits > .5;

  auto accuracy = xt::mean(xt::isclose(actual, predictions));
  IC(accuracy);

  return EXIT_SUCCESS;
}
