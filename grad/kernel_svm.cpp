/**
 *  @file support_vector_machine.cpp
 *  @author Ralph 'Blake' Vente
 *  @license Mozilla Public License
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at https://mozilla.org/MPL/2.0/.
 **/

#include <fstream>
#include <iostream>
#include <string>

#include "icecream.hpp"
#include "xframe/xio.hpp"
#include "xframe/xvariable.hpp"
#include "xtensor-blas/xlinalg.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xcsv.hpp"
#include "xtensor/xrandom.hpp"

using d_vec = xt::xarray<double>;
using t = std::size_t;
using d = double;

// Implements a support vector machine
// @param features training set array<double> with dims (n×m)
// @param target training labels
// @param max_iter int max num iterations
// @param learning_rate (η or α) double
// @return weights 1-d vector result
struct SVM {
  d_vec weights, features, target;
  t max_iter;
  d learning_rate;
  SVM(const d_vec& features, const d_vec& target, const t max_iter,
      const d learning_rate)
      : weights(xt::zeros<d>({features.shape()[1]})),
        features(features),
        max_iter(max_iter),
        target(target),
        learning_rate(learning_rate) {}
  d_vec kernelize(const d_vec& features_j) const {
    // return xt::pow(xt::linalg::dot(this->weights, features_j), 2);
    // return xt::linalg::dot(this->weights, features_j);
  }
  d_vec train() {
    IC(features, weights, target);
    for (t i = 0; i < max_iter; i++) {
      for (t j = 0; j < features.shape()[0]; ++j) {
        auto features_j = xt::view(features, j, xt::all());
        auto target_j = xt::view(target, j, xt::all());
        auto pred = target_j * this->kernelize(features_j);

        auto regularization = -2 * 1.0 / float(max_iter) * weights;
        if (xt::any(xt::less_equal(pred, d_vec(1)))) {
          weights =
              weights + learning_rate * (target_j * features_j) - regularization;
        } else {
          weights = weights + learning_rate * regularization;
        }
      }
      if (!(i % 1000)) {
        IC(i, weights);
      }
    }
    return weights;
  }
  d_vec predict(const d_vec& test_set) {
    IC();
    d_vec predictions = this->kernelize(xt::transpose(test_set));
    IC();
    return predictions;
  }
};

int main(const int argc, const char* argv[]) {
  icecream::ic.prefix("DEBUG| ").line_wrap_width(80);
  const int N_ITER = 1000;
  const double LEARNING_RATE = .000001;

  // create two classes A and B which are normally distributed
  auto A = xt::random::randn<double>({100, 3}, 100, .01);
  auto B = xt::random::randn<double>({100, 3}, -100, .01);

  // concatenate
  auto features = xt::vstack(xt::xtuple(A, B));

  auto labels = xt::squeeze(
      xt::hstack(xt::xtuple(xt::ones<d>({1, 100}), xt::zeros<d>({1, 100}))));
  auto e_features = xt::eval(features);
  xt::random::seed(0);
  xt::random::shuffle(e_features);

  auto e_labels = xt::eval(labels);
  xt::random::seed(0);
  xt::random::shuffle(e_labels);

  auto A_test = xt::random::randn<double>({20, 3}, 100, .1);
  auto B_test = xt::random::randn<double>({20, 3}, -100, .1);
  auto test_set = xt::vstack(xt::xtuple(A_test, B_test));

  auto clf = SVM(e_features, e_labels, N_ITER, LEARNING_RATE);
  clf.train();
  IC(clf.predict(A_test));
  IC(clf.predict(B_test));

  return EXIT_SUCCESS;
}
