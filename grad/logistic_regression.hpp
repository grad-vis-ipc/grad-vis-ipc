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

using d_vec = xt::xarray<double>;
using t = std::size_t;
using d = double;
d_vec logistic_regression(const d_vec & features, const d_vec & target, t num_steps, d learning_rate);