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
d_vec support_vector_machine(const d_vec& features, const d_vec& target,
                             const t max_iter, const d learning_rate) {
  d_vec weights = xt::zeros<d>({features.shape()[1]});
  IC(features, weights, target);
  for (t i = 0; i < max_iter; i++) {
    for (t j = 0; j < features.shape()[0]; ++j) {
      auto features_j = xt::view(features, j, xt::all());
      auto target_j = xt::view(target, j, xt::all());
      // forward pass
      auto pred = target_j * xt::linalg::dot(weights, features_j);
      
      auto regularization = -2 * 1.0 / float(max_iter) * weights;
      // IC(pred);
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