#include "kernels.cpp"
#include "stats.cpp"
#include "support_vector_machine.hpp"
#include "kernel_svm.cpp"
#include "xtensor/xadapt.hpp"

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

  // auto svm_train_labels = 2 * train_labels - xt::ones_like(train_labels);
  // auto svm_train_set = train_set;

  // d_vec predictions = xt::linalg::dot(xt::eval(test_set), svm_weights);
  // IC(predictions);

  // d_vec actual = xt::eval(test_labels);
  // auto accuracy = xt::mean(xt::isclose(actual, predictions));
  // IC(accuracy);

  auto svm_train_labels = xt::xarray<double>({-1, -1, 1, 1});
  // auto svm_train_set = xt::xarray<double>({{1, -1}, {0, 0}, {1, 1}, {2, 0}});
  auto svm_train_set = xt::xarray<double>({{1, -1.5}, {0, -.5}, {1, .5}, {2, -1.5}});

  auto svm_weights = support_vector_machine(svm_train_set, svm_train_labels,
                                            N_ITER, LEARNING_RATE);

  IC(svm_weights);

  d_vec predictions = xt::linalg::dot(d_vec({{2, 2}, {-1, -1}}), svm_weights);
  IC(predictions);

  return EXIT_SUCCESS;
}
