/**
 *  @file logistic_regression.cpp
 *  @author Ralph 'Blake' Vente
 *  @license Mozilla Public License
 *
 *  depends:  xtensor xframe xtensor-blas
 *  AND `apt install libblas-dev liblapack-dev`
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

// Implements logistic regresion
// @param features training set array<double> with dims (n×m)
// @param target training labels
// @param max_iter int max num iterations
// @param learning_rate (η or α) double
// @return weights 1-d vector result
d_vec logistic_regression(const d_vec& features, const d_vec& target,
                          const t max_iter, const d learning_rate) {
  d_vec weights = xt::zeros<d>({features.shape()[1]});
  // for early stopping
  d_vec ll_old(1e99);
  for (t i = 0; i < max_iter; ++i) {
    // initial dot product before sigmoid
    auto scores = xt::linalg::dot(features, weights);

    // pipe inputs through a sigmoid
    auto predictions = 1.0 / (1.0 + xt::exp(-1 * scores));

    // acquire vector of errors made
    auto error = target - predictions;

    // compute the gradient based on error
    auto gradient = xt::linalg::dot(xt::transpose(features), error);

    // update step
    weights = weights + gradient * learning_rate;

    if (i % 1000 == 0) {
      // log likelihood should change between iterations
      auto ll = xt::sum(target * (scores)-xt::log(1 + xt::exp(scores)));
      std::cout << i << " " << ll << std::endl;
      if (xt::allclose(ll, ll_old)) {
        std::cout << "Converged in ≈" << i << " iterations" << std::endl;
        break;
      } else if (xt::any(xt::greater_equal(ll, ll_old))) {
        std::cout << "Diverged in ≈" << i << " iterations" << std::endl;
        break;
      }
      ll_old = ll;
    }
  }
  return weights;
}
