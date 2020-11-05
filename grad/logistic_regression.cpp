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

#include <iostream>
#include <fstream>
#include <string>
#include "xtensor-blas/xlinalg.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor/xcsv.hpp"
#include "xtensor/xarray.hpp"
#include "xframe/xio.hpp"
#include "xframe/xvariable.hpp"
#include "xtensor/xcsv.hpp"
#include "icecream.hpp"

using dvec = xt::xarray<double>;
using t = std::size_t;
using d = double;

// Implements logistic regresion
// @param features training set array<double> with dims (n×m)
// @param target training labels
// @param max_iter int max num iterations 
// @param learning rate (η or α) double
// @return weights (1×m) result
auto logistic_regression(const dvec & features, const dvec & target, t max_iter, d learning_rate) {
  xt::xarray<d> weights = xt::zeros<d>({features.shape()[1]});
  // for early stopping
  xt::xarray<d> ll_old(1e99);
  for (t i = 0; i < max_iter; ++i) {
    // initial dot product
    auto scores = xt::linalg::dot(features, weights);

    // pipe inputs through a sigmoid to get ther error
    auto predictions = 1.0/(1.0 + xt::exp(-1 * scores));

    // compute the gradient
    auto gradient = xt::linalg::dot(xt::transpose(features),(target-predictions));

    // update step
    weights = weights + gradient*learning_rate;

    if (i % 1000 == 0) {
      // log likelihood should decrease
      auto ll = xt::sum(target*(scores) - xt::log(1 + xt::exp(scores)));
      std::cout << i << " " << ll << std::endl;
      if ( xt::allclose(ll, ll_old)) {
        std::cout << "Early stop" << std::endl;
        break;
      }
      ll_old = ll;
    }
  }
  return weights;
}
