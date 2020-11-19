#include "support_vector_machine.hpp"
#include "xtensor/xadapt.hpp"
#include "kernels.cpp"
#include "stats.cpp"

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
  auto test_set = xt::eval(
      xt::view(test_set_raw, xt::all(), xt::range(1, xt::placeholders::_)));

  auto kernel_train = kern::pearson_r(train_set, train_labels);

  auto svm_weights =
      support_vector_machine(kernel_train, train_labels, N_ITER, LEARNING_RATE);

  // IC(kernel_train, kernel_train.shape());

  // use weights to generate predictions
  auto scores = xt::linalg::dot(test_set, svm_weights);
  b_vec predictions = scores > 0;

  auto accuracy = xt::mean(xt::isclose(test_labels, predictions));
  IC(accuracy);

  return EXIT_SUCCESS;
}
