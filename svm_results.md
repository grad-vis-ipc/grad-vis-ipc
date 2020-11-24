vente@osiris:~/repos/grad-vis-ipc/grad$ make svm MX_ITER=10000
./svm.out data_monks/monks-1.train data_monks/monks-1.test 10000 .001
ic| features: {{ 1. , -1.5},
 { 0. , -0.5},
 { 1. ,  0.5},
 { 2. , -0.5}}
    weights: { 0.,  0.}
    target: {-1., -1.,  1.,  1.}
ic| svm_weights: { 1.002231,  2.00068 }
ic| predictions: { 6.005823, -3.002911}
vente@osiris:~/repos/grad-vis-ipc/grad$ make svm MX_ITER=100000
./svm.out data_monks/monks-1.train data_monks/monks-1.test 100000 .001
ic| features: { { 1. , -1.5},
                { 0. , -0.5},
                { 1. ,  0.5},
                { 2. , -0.5}}
    weights: { 0.,  0.}
    target: {-1., -1.,  1.,  1.}
ic| svm_weights: { 1.001843,  2.000311}
ic| predictions: { 6.004309, -3.002155}
ic| i: 0, weights: { 0.002   ,  0.002   }
ic| i: 1000, weights: { 0.845292,  1.380736}
ic| i: 2000, weights: { 0.972405,  1.888062}
ic| i: 3000, weights: { 1.00089 ,  2.000392}
ic| i: 4000, weights: { 1.00081 ,  2.000232}
ic| i: 5000, weights: { 1.00073 ,  2.000072}
ic| i: 6000, weights: { 1.000669,  2.000452}
ic| i: 7000, weights: { 1.000589,  2.000292}
ic| i: 8000, weights: { 1.000509,  2.000132}
ic| i: 9000, weights: { 1.000449,  2.000512}
ic| i: 10000, weights: { 1.000369,  2.000352}
ic| i: 11000, weights: { 1.000289,  2.000192}
ic| i: 12000, weights: { 1.000209,  2.000032}
ic| i: 13000, weights: { 1.000149,  2.000412}
ic| i: 14000, weights: { 1.000069,  2.000252}
ic| i: 15000, weights: { 1.002029,  2.000172}
ic| i: 16000, weights: { 1.001949,  2.000012}
ic| i: 17000, weights: { 1.001889,  2.000392}
ic| i: 18000, weights: { 1.001809,  2.000232}
ic| i: 19000, weights: { 1.001729,  2.000072}
ic| i: 20000, weights: { 1.001669,  2.000452}
ic| i: 21000, weights: { 1.001588,  2.000292}
ic| i: 22000, weights: { 1.001508,  2.000132}
ic| i: 23000, weights: { 1.001448,  2.000512}
ic| i: 24000, weights: { 1.001368,  2.000352}
ic| i: 25000, weights: { 1.001288,  2.000192}
ic| i: 26000, weights: { 1.001208,  2.000032}
ic| i: 27000, weights: { 1.001148,  2.000412}
ic| i: 28000, weights: { 1.001068,  2.000252}
ic| i: 29000, weights: { 1.000988,  2.000092}
ic| i: 30000, weights: { 1.000928,  2.000472}
ic| i: 31000, weights: { 1.000848,  2.000312}
ic| i: 32000, weights: { 1.000768,  2.000152}
ic| i: 33000, weights: { 1.000708,  2.000532}
ic| i: 34000, weights: { 1.000628,  2.000372}
ic| i: 35000, weights: { 1.000547,  2.000212}
ic| i: 36000, weights: { 1.000467,  2.000052}
ic| i: 37000, weights: { 1.000407,  2.000432}
ic| i: 38000, weights: { 1.000327,  2.000272}
ic| i: 39000, weights: { 1.000247,  2.000112}
ic| i: 40000, weights: { 1.000187,  2.000492}
ic| i: 41000, weights: { 1.000107,  2.000332}
ic| i: 42000, weights: { 1.002067,  2.000252}
ic| i: 43000, weights: { 1.001987,  2.000092}
ic| i: 44000, weights: { 1.001927,  2.000472}
ic| i: 45000, weights: { 1.001847,  2.000312}
ic| i: 46000, weights: { 1.001767,  2.000152}
ic| i: 47000, weights: { 1.001707,  2.000532}
ic| i: 48000, weights: { 1.001627,  2.000372}
ic| i: 49000, weights: { 1.001546,  2.000212}
ic| i: 50000, weights: { 1.001466,  2.000052}
ic| i: 51000, weights: { 1.001406,  2.000432}
ic| i: 52000, weights: { 1.001326,  2.000272}
ic| i: 53000, weights: { 1.001246,  2.000112}