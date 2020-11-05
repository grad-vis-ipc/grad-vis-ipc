#include "logistic_regression.hpp"
#include "xtensor/xadapt.hpp"

int main(const int argc, const char* argv[])
{
  if (argc < 5) {
    std::cout << "Usage: " << argv[0] << " <train> <test> <iterations> <step>" << std::endl;
    return EXIT_FAILURE;
  }
  const auto [
    train_fname, test_fname, N_ITER, LEARNING_RATE
    ] = std::make_tuple(argv[1], argv[2], atoi(argv[3]), std::stod(argv[4]));

  std::ifstream train(train_fname);
  auto training_set_raw = xt::load_csv<double>(train);
  train.close();

  std::ifstream test(test_fname);
  auto test_set_raw = xt::load_csv<double>(test);
  test.close();

  auto train_labels = xt::col(training_set_raw, 0);
  auto training_set = xt::view(training_set_raw, xt::all(), xt::range(1,xt::placeholders::_) );

  auto test_labels = xt::col(test_set_raw, 0);
  auto test_set = xt::view(test_set_raw, xt::all(), xt::range(1,xt::placeholders::_) );

  IC(train_labels, training_set);
  auto weights  = logistic_regression(training_set, train_labels, N_ITER, LEARNING_RATE);
  IC(weights);
  auto scores = xt::linalg::dot(xt::eval(test_set), weights);

  auto predictions = 1.0/(1.0 + xt::exp(-1 * scores));
  auto a  = xt::eval(test_labels);
  auto p =  xt::eval(predictions > .5);

  std::vector<double> actual(a.begin(), a.end());
  std::vector<bool> bool_predictions_vec(p.begin(), p.end());
  double sum = 0;
  for (int i = 0; i < p.size(); ++i) {
    sum += (actual[i] == p[i]);
  }
  auto accuracy = sum/p.size();
  IC(accuracy);

  return EXIT_SUCCESS;
}