# DRGP-VSS
# (licenced under MIT licence)

The code is split into three parts:

1. matlab initialization of the data for python
2. python optimization/training
3. matlab simulation of the model

Steps to go:

1. - in \DRGP_VSS\matlab open the calibrate.m and change your prefered setting
   - choose create_data_python = 'on'; simulation = 'off'; to create the preprocessed data for python
   - Run STARTME.m
   - the created file 'load_python.mat' was saved in \DRGP_VSS\python\data
2. - in \DRGP_VSS\python open 'DEEPvSSGP_STARTME_SV1.py' (DRGP-VVS) or 'DEEPvSSGP_STARTME_SV2.py' (DRGP-SS)
   - choose a specify filename to load (# specify filename to load --> load_python.mat or other)
   - if you have chosen non_rec = 'on' in matlab, specify non_rec = 1; in python, otherwise 0
   - choose the to optimize variables and iterations
   - Run 'DEEPvSSGP_STARTME_SV1.py' (DRGP-VVS) or 'DEEPvSSGP_STARTME_SV2.py' (DRGP-SS)
   - for the first time theano will compile the code
   - the optimized file is saved in \DRGP_VSSmatlab\data_optimized
3. - in \DRGP_VSS\matlab open the calibrate.m and choose the same setting as in 1. unless
     choose create_data_python = 'off'; simulation = 'on' and specify
     filename_optimized_parameters = '...';
   - Run STARTME.m and wait for the results
   
We also offer some of our achieved results in \DRGP_VSS\matlab\data_optimized.
The raw data is available in \DRGP_VSS\matlab\data_raw.

Our code relies on some code snippets of VSSGP (https://github.com/yaringal/VSSGP).
This means we took just these parts of the code which brings our new algorithm/method to work with theano/python.
The main algorithm is new.

For these code snippets we include:

Copyright 2015 Yarin Gal

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

