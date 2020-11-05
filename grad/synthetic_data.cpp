#include "logistic_regression.hpp"
#include "xtensor/xadapt.hpp"


int main(int argc, char* argv[])
{
  // create two classes A and B which are normally distributed with means of 4 and -4
  auto A = xt::random::randn<double>({100, 2}, 4, .1);
  auto B = xt::random::randn<double>({100, 2}, -4, .1);

  // concatenate 
  auto features = xt::vstack(xt::xtuple(A,B));

  // std::cout << features << std::endl;
  auto labels = xt::squeeze(
    xt::hstack(
      xt::xtuple(xt::ones<d>({1,100}), xt::zeros<d>({1,100}))
      ));
  // std::cout << labels << std::endl;
  IC(labels);
  auto e_features = xt::eval(features);
  xt::random::seed(0);
  xt::random::shuffle(e_features);

  auto e_labels = xt::eval(labels);
  xt::random::seed(0);
  xt::random::shuffle(e_labels);

  auto weights  = logistic_regression(e_features, e_labels, 8000, 5e-3);
  IC(weights);

  auto scores = xt::linalg::dot(features, weights);
  auto predictions = 1.0/(1.0 + xt::exp(-1 * scores));
  IC(predictions);
  IC(predictions > .5);

  {
    auto scores = xt::linalg::dot(e_features, weights);
    auto predictions = 1.0/(1.0 + xt::exp(-1 * scores));
    IC(predictions);
    IC(predictions > .5);
  }
  return 0;
}