#!/usr/bin/env python3

import argparse
import arviz as az
import csv
import engineering_notation
import hashlib
import numpy as np
import os
import pandas as pd
import platform
import pymc3 as pm
import shutil
import scipy.stats
import statistics
import subprocess
import sys
import time
import tqdm
import warnings


# Miscellaneous functions
def eng(x):
    return engineering_notation.EngNumber(x, precision=3)


def delay(delay_var, delay_name):
    if delay_var > 0:
        if delay_name is not None:
            print('Sleeping between {} for {}s'.format(delay_name, delay_var),
                  flush=True)
        time.sleep(delay_var)


def save_to_csv(filename, num_runs, tests, baseline_src, files, serial_time,
                parallel_time):
    all_files = [baseline_src] + files
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')

        writer.writerow(['Test', 'Run', 'Serial'] + all_files)

        for test in tests:
            for run in range(num_runs):
                row = [test, run+1, serial_time[test][run]]

                for f in all_files:
                    row += [parallel_time[test][f][run]]

                writer.writerow(row)


# Compilation
def generate_makefile(generator, c_compiler, cxx_compiler):
    shutil.rmtree('build', ignore_errors=True)
    os.mkdir('build')
    os.chdir('build')
    compiler_string = ''
    if c_compiler is not None:
        compiler_string += '-DCMAKE_C_COMPILER=' + c_compiler + ' '
    if cxx_compiler is not None:
        compiler_string += '-DCMAKE_CXX_COMPILER=' + cxx_compiler + ' '
    if c_flags is not None:
        compiler_string += '-DCMAKE_C_FLAGS=' + c_flags + ' '
    if cxx_flags is not None:
        compiler_string += '-DCMAKE_CXX_FLAGS=' + cxx_flags + ' '
    os.system('cmake ' + generator + compiler_string +
              '-DCMAKE_BUILD_TYPE=Release ..')
    os.chdir('..')


def compile_code():
    # Cleaning is required because cmake sometimes doesn't recognize changes
    # after copying a new version of the source file in compile_all().
    subprocess.run(['cmake', '--build', 'build', '--target', 'clean'])
    r = subprocess.run(['cmake', '--build', 'build', '--config', 'Release'])
    if r.returncode != 0:
        sys.stderr.write("Compile error\n")
        sys.exit(1)


def compile_all(executable_path, parallel_executable, executable_extension,
                parallel_extension, parallel_src, baseline_src, files):
    shutil.copy(os.path.join('src', baseline_src + parallel_extension),
                os.path.join('src', parallel_src + parallel_extension))
    compile_code()
    shutil.copy(os.path.join(executable_path,
                             parallel_executable + executable_extension),
                os.path.join(executable_path,
                             baseline_src + executable_extension))

    for f in files:
        shutil.copy(os.path.join('src', f + parallel_extension),
                    os.path.join('src', parallel_src + parallel_extension))
        compile_code()
        shutil.copy(os.path.join(executable_path,
                                 parallel_executable + executable_extension),
                    os.path.join(executable_path, f + executable_extension))


# Statistical functions
def statistical_analysis(baseline, opt):
    # Inspired by https://docs.pymc.io/notebooks/BEST.html
    y = pd.DataFrame(
        dict(
            value=np.r_[baseline, opt],
            group=np.r_[['baseline']*len(baseline), ['opt']*len(opt)]
        )
    )

    μ_m = y.value.mean()
    μ_s = y.value.std()
    σ_low = µ_s/1000
    σ_high = µ_s*1000

    with pm.Model() as model:
        baseline_mean = pm.Normal('baseline_mean', mu=μ_m, sd=1000*μ_s)
        opt_mean = pm.Normal('opt_mean', mu=μ_m, sd=1000*μ_s)
        baseline_std = pm.Uniform('baseline_std', lower=µ_s/1000,
                                  upper=1000*µ_s)
        opt_std = pm.Uniform('opt_std', lower=µ_s/1000, upper=1000*µ_s)
        ν = pm.Exponential('ν_minus_one', 1/29.) + 1
        λ_baseline = baseline_std**-2
        λ_opt = opt_std**-2

        dist_baseline = pm.StudentT('baseline', nu=ν, mu=baseline_mean,
                                    lam=λ_baseline, observed=baseline)
        dist_opt = pm.StudentT('opt', nu=ν, mu=opt_mean,
                               lam=λ_opt, observed=opt)

        diff_of_means = pm.Deterministic('difference of means',
                                         baseline_mean - opt_mean)
        ratio_of_means = pm.Deterministic('ratio of means',
                                          baseline_mean/opt_mean)

        trace = pm.sample(draws=3000, tune=2000)

        baseline_hdi = az.hdi(trace['baseline_mean'])
        baseline_out = (baseline_hdi[0],
                        trace['baseline_mean'].mean(),
                        baseline_hdi[1])

        opt_hdi = az.hdi(trace['opt_mean'])
        opt_out = (opt_hdi[0], trace['opt_mean'].mean(), opt_hdi[1])

        speedup_hdi = az.hdi(trace['ratio of means'])
        speedup = (speedup_hdi[0],
                   trace['ratio of means'].mean(),
                   speedup_hdi[1])

        dif = trace['difference of means'] > 0
        prob = (dif > 0).sum()/len(dif)

    return (baseline_out, opt_out, speedup, prob)


def compute_all_statistics(tests, baseline_src, files, serial_time,
                           parallel_time):
    serial_hdi = {}
    bl_hdi = {}
    bl_speedup_hdi = {}
    bl_prob_hdi = {}
    bl_hdi2 = {}
    opt_hdi = {}
    opt_improv_hdi = {}
    opt_prob_hdi = {}

    for test in tests:
        print('\nComputing baseline statistics for test {}'.format(test))
        (serial_hdi[test],
         bl_hdi[test],
         bl_speedup_hdi[test],
         bl_prob_hdi[test]) = statistical_analysis(
             serial_time[test], parallel_time[test][baseline_src])

        bl_hdi2[test] = {}
        opt_hdi[test] = {}
        opt_improv_hdi[test] = {}
        opt_prob_hdi[test] = {}

        for f in files:
            print("\nComputing statistics for file '{}', "
                  "test {}".format(f, test))
            (bl_hdi2[test][f],
             opt_hdi[test][f],
             opt_improv_hdi[test][f],
             opt_prob_hdi[test][f]) = statistical_analysis(
                parallel_time[test][baseline_src], parallel_time[test][f]
            )

    return (serial_hdi, bl_hdi, bl_speedup_hdi, bl_prob_hdi,
            bl_hdi2, opt_hdi, opt_improv_hdi, opt_prob_hdi)


def compute_and_print_statistics(tests, baseline_src, files, serial_time,
                                 parallel_time):
    print('\nComputing statistics, this may take a while')

    (serial_hdi, bl_hdi, bl_speedup_hdi, bl_prob_hdi,
     bl_hdi2, opt_hdi, opt_improv_hdi, opt_prob_hdi) = compute_all_statistics(
        tests, baseline_src, files, serial_time, parallel_time)

    print('\nStatistics:')
    print('Baseline:')
    for test in tests:
        print(
            '\tTest {}:\n'
            '\t\tavg speedup = {:.4f}x HDI = ({:.4f}x,{:.4f}x)\n'
            '\t\tavg tser = {}s HDI = ({},{})\n'
            '\t\tavg tpar = {}s HDI = ({},{})\n'
            '\t\tP(tpar < tser) = {:.1f}%'.format(
                test,
                bl_speedup_hdi[test][1],
                bl_speedup_hdi[test][0],
                bl_speedup_hdi[test][2],
                eng(serial_hdi[test][1]),
                eng(serial_hdi[test][0]),
                eng(serial_hdi[test][2]),
                eng(bl_hdi[test][1]),
                eng(bl_hdi[test][0]),
                eng(bl_hdi[test][2]),
                100*bl_prob_hdi[test]
            )
        )

    for f in files:
        print('File {}:'.format(f))
        for test in tests:
            print(
                '\tTest {}:\n'
                '\t\tavg improvement = {:.4f}x HDI = ({:.4f}x,{:.4f}x)\n'
                '\t\tavg tbl  = {}s HDI = ({},{})\n'
                '\t\tavg topt = {}s HDI = ({},{})\n'
                '\t\tP(topt < tbl) = {:.1f}%'.format(
                    test,
                    opt_improv_hdi[test][f][1],
                    opt_improv_hdi[test][f][0],
                    opt_improv_hdi[test][f][2],
                    eng(bl_hdi2[test][f][1]),
                    eng(bl_hdi2[test][f][0]),
                    eng(bl_hdi2[test][f][2]),
                    eng(opt_hdi[test][f][1]),
                    eng(opt_hdi[test][f][0]),
                    eng(opt_hdi[test][f][2]),
                    100*opt_prob_hdi[test][f]
                )
            )


# Test runners
def run_one_test(file_path, test_path, pbar=None, pbar_value=None):
    t = subprocess.run(
        [file_path, test_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output_hash = hashlib.sha256(t.stdout).hexdigest()
    exec_time = float(t.stderr)
    # Avoid division by zero if the test is faster than 1 us
    if exec_time < 1e-6:
        exec_time = 1e-6

    return (exec_time, output_hash)


def run_all_tests(num_runs, tests, executable_path, executable_extension,
                  parallel_src, baseline_src, files, delay_run=None,
                  initial_run=True):
    serial_hash = {}
    serial_time = {}
    parallel_time = {}

    all_files = [baseline_src] + files

    for test in tests:
        serial_time[test] = []
        serial_hash[test] = []
        parallel_time[test] = {}
        for f in all_files:
            parallel_time[test][f] = []

    if initial_run is True:
        total = len(tests)*(len(all_files) + 1)
    else:
        total = num_runs

    with tqdm.tqdm(total=total,
                   bar_format='{l_bar}{bar}|{elapsed}<{remaining} ') as pbar:
        for _ in range(num_runs):
            for test in tests:
                test_path = os.path.join('tests', test + '.in')
                serial_path = os.path.join(executable_path,
                                           'serial' + executable_extension)

                if delay_run is not None:
                    delay(delay_run[test], None)

                (t, h) = run_one_test(serial_path, test_path)
                if initial_run is True:
                    pbar.update()

                serial_time[test].append(t)
                serial_hash[test] = h

                for f in all_files:
                    parallel_path = os.path.join(executable_path,
                                                 f + executable_extension)
                    if delay_run is not None:
                        delay(delay_run[test], None)

                    (t, h) = run_one_test(parallel_path, test_path)
                    if initial_run is True:
                        pbar.update()

                    if h != serial_hash[test]:
                        sys.stderr.write(
                            'Error in test {}: output of file {} is different '
                            'from serial code\n'.format(test, f))
                        sys.exit(1)

                    parallel_time[test][f].append(t)

            if initial_run is False:
                pbar.update()

    return (serial_time, parallel_time)


# HOW TO USE THIS SCRIPT
#
# Save it in the folder that contains CMakeLists.txt.
#
# Install required packages using pip: arviz, engineering_notation, pandas,
# pymc3, numpy, scipy and tqdm.
#
# Save source files that you want to test in the src/ folder. Choose one
# to be the baseline result, the one to be compared against other (presumably
# optimized) versions. The baseline file name should be placed in the
# 'baseline_src' variable below, whereas the list of optimized files should
# be placed in the 'files' variable.
#
# Go through the definitions below, checking if they make sense. In
# principle the file names ('parallel_src', 'baseline_src' and 'files') need
# to be changed from one exercise to the next, and you should select the
# appropriate compilers for your environment, in variables 'c_compiler' and
# 'cxx_compiler' (the default setting of None allows CMake to choose the
# compiler automatically).
#
# Finally, just run the script. It should perform all testing automatically
# and report figures at the end.
#
# Note this has been tested on macOS 10.15 and Windows 10 using Cygwin and
# Visual Studio. It may have errors in other platforms; if you fix them,
# please send a pull request.


# HOW TO INTERPRET THE OUTPUT
#
# Here is some sample output:
#
###############################################################################
#
# Statistics:
# Baseline:
#     Test 1:
#         avg speedup = 2.3918x HDI = (2.3911x,2.3925x)
#         avg tser = 206.304ms HDI = (206.261m,206.348m)
#         avg tpar = 86.254ms HDI = (86.237m,86.271m)
#         P(tpar < tser) = 100.0%
#
#         ...
#
#     Test 5:
#         avg speedup = 2.3576x HDI = (2.3571x,2.3581x)
#         avg tser = 40.894s HDI = (40.891,40.897)
#         avg tpar = 17.346s HDI = (17.342,17.349)
#         P(tpar < tser) = 100.0%
# File xxx:
#     Test 1:
#         avg improvement = 1.6256x HDI = (1.6252x,1.6260x)
#         avg tbl  = 86.243ms HDI = (86.236m,86.249m)
#         avg topt = 53.053ms HDI = (53.040m,53.067m)
#         P(topt < tbl) = 100.0%
#
#         ...
#
#     Test 5:
#         avg improvement = 1.6856x HDI = (1.6853x,1.6858x)
#         avg tbl  = 17.345s HDI = (17.344,17.346)
#         avg topt = 10.291s HDI = (10.290,10.292)
#         P(topt < tbl) = 100.0%
###############################################################################
#
# Results are grouped by file (baseline and optimized files) and then by test
# number.
#
# Speedups are given as the ratio of the serial and parallel execution times.
# Improvements between the optimized and baseline parallel version are given
# by the ratio of the baseline parallel execution time by the optimized
# parallel execution time. The execution times are also reported.
#
# Bayesian analysis is performed on the datasets to obtain a point estimate
# (mean of each variable) as well as 95% credible intervals for each. A
# probability that one of the execution times is smaller than the other one is
# reported; if low or high enough (say, 5% or 95%), then it is reasonable to
# assume that a difference actually exists.
#
# Note that intersecting credible intervals, or probabilities in the range
# (5%,95%) may either indicate there is no discernible difference between the
# execution time of the baseline and optimized files, or that the number of
# runs is insufficient to declare that a statistically significant difference
# exists. It's up to you to interpret which one is the case in your runs, and
# if necessary, run the script again increasing the 'num_runs' variable to
# try to achieve a definite answer.


# BEST PRACTICES FOR BENCHMARKING
# For best results, you should close off as many apps/services as possible,
# as anything competing for CPU time will potentially poison the results.
#
# It's also recommended to disable Turbo Boost or equivalent technology in
# your CPU, since clock speeds are usually higher when a single core is in
# use (e.g. when running serial code) and lower when multiple cores are in
# use (e.g. when running parallel code). This means your speedups will be
# too pessimistic. In addition, clock speed may vary throughout the run,
# even if you have a good cooling solution, further poisoning the results.
# See https://bit.ly/2H2eChG for more details on Turbo Boost issues.
# Disabling Turbo Boost is OS-dependent. For macOS, there is an app called
# Turbo Boost Switcher. You may need to search for similar apps for other
# OSes.
#
# When CPUs reach high temperatures (e.g. 100 ºC), they forcefully reduce
# clock speeds to avoid damaging the processor. This poisons the results, and
# if your hardware is under the risk of thermal throttling (especially
# laptops), you should take steps to mitigate the issues.
#
# Disabling Turbo Boost as suggested previously may be enough to solve the
# problem, as the CPU will dissipate much less power.
#
# It's also suggested that you configure your fans to run at full speed all
# the time while testing. They may take too long to ramp up, and reach the
# temperature threshold (and therefore activate thermal throttling).
#
# If, despite previous suggestions, temperatures are still high enough that
# thermal throttling kicks in (even if infrequently), it's suggested that
# you add delays between runs to let your CPU cool off (see the 'delay_run'
# variable below). You'll need to experiment with different values to see
# the minimum required to avoid thermal throttling.
#
# Under Windows, you may want to change the power plan to "High performance"
# (in Control Panel -> Hardware and Sound -> Power Options). This uses more
# power, but keeps the CPU at its nominal clock speed all the time, removing
# a source of timing variability, as the CPU may take some time to "spool up"
# from a lower clock speed (for power savings) to a higher one for
# CPU-intensive workloads. See https://bit.ly/3163kjD for more details.


if __name__ == '__main__':
    # DEFINITIONS

    # How many repeated runs (of all files and tests)
    # Use more if needed to achieve statistical significance
    num_runs = 50

    # Which tests to run (one for each file in the 'tests' folder)
    # Do not add .in extension, it's done automatically
    tests = ['1', '2', '3', '4', '5']

    # Folder for executable files; may be different under
    # Windows when using VS's version of cmake
    executable_path = 'build'  # os.path.join('build', 'Release')

    # Default name for the executable of the parallel code
    # In principle this shouldn't be changed
    parallel_executable = 'parallel'

    # Extension for parallel files, typically either '.c' or '.cu'
    parallel_extension = '.c'

    # Default name for the source file of the parallel code (note this file
    # will be overwritten by the script with the baseline and optimized files)
    # This changes from one exercise to the next
    parallel_src = 'prime-parallel'

    # Name of your baseline file (the one others will be compared against)
    baseline_src = 'prime-parallel-baseline'

    # List of your optimized files to be compared against the baseline file
    files = ['prime-parallel-opt1']

    # Choice of compiler, or None to let CMake choose automatically
    c_compiler = None    # 'gcc-10' # 'icl' # 'x86_64-w64-mingw32-gcc-10'
    cxx_compiler = None  # 'g++-10' # 'icl' # 'x86_64-w64-mingw32-g++'

    # Extra C and C++ compiler flags that may be required for some reason
    # Use None if no extra flags are necessary
    c_flags = None
    cxx_flags = None

    # As explained above, this adds a delay between each run of the code to
    # cool off the CPU and reduce, or ideally prevent, thermal throttling. This
    # is a dict mapping of tests (as in, elements of the 'tests' variable
    # above) to the number of seconds the CPU will sleep after a given test.
    # The default value of None means no delays are inserted
    delay_run = None  # { '1': 0, '2': 0, '3': 1, '4': 3, '5': 6 }

    # Name of a file to save the measurements to (in CSV format), or None
    csv_output_file = None

    # OS-specific definitions
    if platform.system() != 'Windows':
        executable_extension = ''
        generator = ''
    else:
        executable_extension = '.exe'
        generator = ''
        # If using Visual Studio, use the generator line below
        # generator = '-G "NMake Makefiles" '

    # Beginning of the script
    warnings.simplefilter('ignore')

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--configure',
        action='store_true',
        help='rerun CMake configure step (required after switching compilers)'
    )
    args = parser.parse_args()

    if delay_run is None:
        print('{} total runs: {} repetitions on {} executables '
              'for test files {}'.format(
                num_runs*(1 + len(files))*len(tests),
                num_runs,
                1 + len(files),
                tests)
              )
    else:
        print('{} total runs: {} repetitions on {} executables '
              'for test files {}, with {}-{}s delay between runs'.format(
                num_runs*(1 + len(files))*len(tests),
                num_runs,
                1 + len(files),
                tests,
                min(delay_run.values()),
                max(delay_run.values()))
              )

    if args.configure or not os.path.exists('build'):
        print('\nGenerating Makefiles')
        generate_makefile(generator, c_compiler, cxx_compiler)

    print('\nCompiling code')
    compile_all(executable_path, parallel_executable, executable_extension,
                parallel_extension, parallel_src, baseline_src, files)

    print('\nStarting initial run')
    (st, pt) = run_all_tests(
        1, tests, executable_path, executable_extension, parallel_src,
        baseline_src, files, delay_run, initial_run=True
    )
    print('\nStarting main run')
    (serial_time, parallel_time) = run_all_tests(
        num_runs, tests, executable_path, executable_extension,
        parallel_src, baseline_src, files, delay_run, initial_run=False
    )

    compute_and_print_statistics(tests, baseline_src, files, serial_time,
                                 parallel_time)

    if csv_output_file is not None:
        save_to_csv(csv_output_file, num_runs, tests, baseline_src, files,
                    serial_time, parallel_time)

    # TODOs:
    # TODO: less runs if a test takes longer, more runs if it's shorter?
    #       Would that be enough for statistical significance?
    # TODO: run some high-performance code for 10-50ms before running tests,
    #       to "spool up" TurboBoost
    # TODO: select number of runs automatically to achieve statistical
    #       significance
    # TODO: select delay between runs automatically (how?)
    # TODO: measure actual execution time of each test run, and use this to
    #       update the progress bar more frequently (but still somewhat
    #       accurately)
    # TODO: break up the project into different files, this single file is
    #       getting huge
    # TODO: add option for summary and full statistics
