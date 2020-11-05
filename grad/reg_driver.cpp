#include "logistic_regression.hpp"
#include "xtensor/xadapt.hpp"

using d_vec = xt::xarray<double>;
using b_vec = xt::xarray<bool>;
using t = std::size_t;
using d = double;

int main(const int argc, const char* argv[])
{
  if (argc < 5) {
    std::cout << "Usage: " << argv[0] << " <train> <test> <iterations> <step>" << std::endl;
    return EXIT_FAILURE;
  }
  const auto [
    train_fname, test_fname, N_ITER, LEARNING_RATE
    ] = std::make_tuple(argv[1], argv[2], atoi(argv[3]), std::stod(argv[4]));

  // load csv from files
  std::ifstream train(train_fname); // training set
  auto training_set_raw = xt::load_csv<double>(train);
  train.close();

  std::ifstream test(test_fname); // test set
  auto test_set_raw = xt::load_csv<double>(test);
  test.close();

  auto train_labels = xt::col(training_set_raw, 0);
  auto training_set = xt::view(training_set_raw, xt::all(), xt::range(1,xt::placeholders::_) );

  auto test_labels = xt::col(test_set_raw, 0);
  auto test_set = xt::view(test_set_raw, xt::all(), xt::range(1,xt::placeholders::_) );

  IC(train_labels, training_set);
  auto weights  = logistic_regression(training_set, train_labels, N_ITER, LEARNING_RATE);
  IC(weights);
  IC(weights.shape());
  auto scores = xt::linalg::dot(xt::eval(test_set), weights);

  auto logits = 1.0/(1.0 + xt::exp(-1 * scores));
  d_vec actual  = xt::eval(test_labels);
  b_vec predictions =  xt::eval(logits > .5);

  auto accuracy = xt::mean(xt::isclose(actual,predictions));
  IC(accuracy);

  return EXIT_SUCCESS;
}