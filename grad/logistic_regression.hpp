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

/* depends:  xtensor xframe xtensor-blas */
/* apt install libblas-dev liblapack-dev */

using vec = xt::xarray<double>;
using t = std::size_t;
using d = double;
xt::xarray<double> logistic_regression(const vec & features, const vec & target, t num_steps, d learning_rate);