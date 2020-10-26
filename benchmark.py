#!/usr/bin/env python3

import argparse
import arviz as az
import copy
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
def statistical_analysis(serial, bl, opt):
    # Inspired by https://docs.pymc.io/notebooks/BEST.html
    value = np.r_[serial, bl]
    group = np.r_[['serial']*len(serial), ['bl']*len(bl)]
    for f in opt:
        value = np.r_[value, opt[f]]
        group = np.r_[group, [f]*len(opt[f])]

    y = pd.DataFrame(dict(value=value, group=group))

    μ_m = y.value.mean()
    μ_s = y.value.std()
    σ_low = µ_s/1000
    σ_high = µ_s*1000

    with pm.Model() as model:
        serial_mean = pm.Normal('serial_mean', mu=µ_m, sd=1000*µ_s)
        serial_std = pm.Uniform('serial_std', lower=µ_s/1000, upper=1000*µ_s)
        λ_serial = serial_std**-2

        bl_mean = pm.Normal('bl_mean', mu=μ_m, sd=1000*μ_s)
        bl_std = pm.Uniform('bl_std', lower=µ_s/1000, upper=1000*µ_s)
        λ_bl = bl_std**-2

        opt_mean = {}
        opt_std = {}
        λ_opt = {}

        for f in opt:
            opt_mean[f] = pm.Normal('opt_{}_mean'.format(f),
                                    mu=μ_m, sd=1000*μ_s)
            opt_std[f] = pm.Uniform('opt_{}_std'.format(f),
                                    lower=µ_s/1000, upper=1000*µ_s)
            λ_opt[f] = opt_std[f]**-2

        ν = pm.Exponential('ν_minus_one', 1/29.) + 1

        dist_serial = pm.StudentT('serial', nu=ν, mu=serial_mean,
                                  lam=λ_serial, observed=serial)
        dist_bl = pm.StudentT('bl', nu=ν, mu=bl_mean, lam=λ_bl, observed=bl)

        dist_opt = {}
        for f in opt:
            dist_opt[f] = pm.StudentT('opt_{}'.format(f), nu=ν, mu=opt_mean[f],
                                      lam=λ_opt[f], observed=opt[f])

        dmean_serial_bl = pm.Deterministic('dmean_serial_bl',
                                           serial_mean - bl_mean)

        dmean_bl_opt = {}
        for f in opt:
            dmean_bl_opt[f] = pm.Deterministic('dmean_bl_opt_{}'.format(f),
                                               bl_mean - opt_mean[f])

        speedup_bl = pm.Deterministic('speedup_bl', serial_mean/bl_mean)

        speedup_opt = {}
        improv_opt = {}
        for f in opt:
            speedup_opt = pm.Deterministic('speedup_opt_{}'.format(f),
                                           serial_mean/opt_mean[f])
            improv_opt = pm.Deterministic('improv_opt_{}'.format(f),
                                          bl_mean/opt_mean[f])

        trace = pm.sample(draws=3000, tune=2000)

        res1 = [('serial', 'serial_mean'), ('bl', 'bl_mean')]
        res2 = [('bl', 'speedup_bl')]
        res3 = []
        res4 = [('bl', 'dmean_serial_bl')]

        for f in opt:
            res1 += [('opt_{}'.format(f), 'opt_{}_mean'.format(f))]
            res2 += [('opt_{}'.format(f), 'speedup_opt_{}'.format(f))]
            res3 += [('opt_{}'.format(f), 'improv_opt_{}'.format(f))]
            res4 += [('opt_{}'.format(f), 'dmean_bl_opt_{}'.format(f))]

        runtime = {}
        for r in res1:
            tr = trace[r[1]]
            hdi = az.hdi(tr)
            runtime[r[0]] = (hdi[0], tr.mean(), hdi[1])

        speedup = {}
        for r in res2:
            tr = trace[r[1]]
            hdi = az.hdi(tr)
            speedup[r[0]] = (hdi[0], tr.mean(), hdi[1])

        improv = {}
        for r in res3:
            tr = trace[r[1]]
            hdi = az.hdi(tr)
            improv[r[0]] = (hdi[0], tr.mean(), hdi[1])

        prob = {}
        for r in res4:
            tr = trace[r[1]]
            prob[r[0]] = (tr > 0).sum()/len(tr)

    return (runtime, speedup, improv, prob)


def compute_all_statistics(tests, baseline_src, files, serial_time,
                           parallel_time):
    runtime = {}
    speedup = {}
    improv = {}
    prob = {}
    for test in tests:
        print('\nComputing statistics for test {}'.format(test))

        parallel_time_opt = dict(parallel_time[test])
        del parallel_time_opt[baseline_src]

        (runtime[test],
         speedup[test],
         improv[test],
         prob[test]) = statistical_analysis(serial_time[test],
                                            parallel_time[test][baseline_src],
                                            parallel_time_opt)

    return (runtime, speedup, improv, prob)


def compute_and_print_statistics(tests, baseline_src, files, serial_time,
                                 parallel_time):
    print('\nComputing statistics, this may take a while')

    (runtime, speedup, improv, prob) = compute_all_statistics(
        tests, baseline_src, files, serial_time, parallel_time)

    print('\nStatistics:')
    print('\tBaseline:')
    for test in tests:
        print(
            '\t\tTest {}:\n'
            '\t\t\tavg speedup = {:.4f}x HDI = ({:.4f}x,{:.4f}x)\n'
            '\t\t\tavg tser = {}s HDI = ({},{})\n'
            '\t\t\tavg tpar = {}s HDI = ({},{})\n'
            '\t\t\tP(tpar < tser) = {:.1f}%'.format(
                test,
                speedup[test]['bl'][1],
                speedup[test]['bl'][0],
                speedup[test]['bl'][2],
                eng(runtime[test]['serial'][1]),
                eng(runtime[test]['serial'][0]),
                eng(runtime[test]['serial'][2]),
                eng(runtime[test]['bl'][1]),
                eng(runtime[test]['bl'][0]),
                eng(runtime[test]['bl'][2]),
                100*prob[test]['bl']
            )
        )

    for f in files:
        print('\tFile {}:'.format(f))
        for test in tests:
            key = 'opt_{}'.format(f)
            print(
                '\t\tTest {}:\n'
                '\t\t\tavg speedup = {:.4f}x HDI = ({:.4f}x,{:.4f}x)\n'
                '\t\t\tavg improvement over baseline = {:.4f}x '
                'HDI = ({:.4f}x,{:.4f}x)\n'
                '\t\t\tavg topt = {}s HDI = ({},{})\n'
                '\t\t\tP(topt < tbl) = {:.1f}%'.format(
                    test,
                    speedup[test][key][1],
                    speedup[test][key][0],
                    speedup[test][key][2],
                    improv[test][key][1],
                    improv[test][key][0],
                    improv[test][key][2],
                    eng(runtime[test][key][1]),
                    eng(runtime[test][key][0]),
                    eng(runtime[test][key][2]),
                    100*prob[test][key]
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
# 	Baseline:
# 		Test 1:
# 			avg speedup = 2.3907x HDI = (2.3897x,2.3918x)
# 			avg tser = 206.216ms HDI = (206.156m,206.280m)
# 			avg tpar = 86.257ms HDI = (86.231m,86.284m)
# 			P(tpar < tser) = 100.0%
#
#       ...
#
# 		Test 5:
# 			avg speedup = 2.3575x HDI = (2.3568x,2.3581x)
# 			avg tser = 40.891s HDI = (40.887,40.895)
# 			avg tpar = 17.345s HDI = (17.341,17.349)
# 			P(tpar < tser) = 100.0%
# 	File xxx:
# 		Test 1:
# 			avg speedup = 3.885661x HDI = (3.8836x,3.8879x)
# 			avg improvement over baseline = 1.6253x HDI = (1.6244x,1.6262x)
# 			avg topt = 53.071ms HDI = (53.048m,53.096m)
# 			P(topt < tbl) = 100.0%
#
#       ...
#
#  		Test 5:
# 			avg speedup = 3.973781x HDI = (3.9719x,3.9756x)
# 			avg improvement over baseline = 1.6856x HDI = (1.6847x,1.6864x)
# 			avg topt = 10.290s HDI = (10.286,10.295)
# 			P(topt < tbl) = 100.0%
###############################################################################
#
# Results are grouped by file (baseline and optimized files) and then by test
# number.
#
# Speedups are given as the ratio of the serial and parallel execution times.
# Improvements of the optimized version over the baseline parallel version are
# given by the ratio of the baseline parallel execution time by the optimized
# parallel execution time. The execution times of each version are also
# reported.
#
# Bayesian analysis is performed on the datasets to obtain a point estimate
# (mean of each variable) as well as 95% credible intervals for each -- i.e.
# the real mean has 95% chance of lying within this interval. Also reported is
# the probability that the execution time for one version is smaller than the
# execution time for another one; if low or high enough (say, < 5% or > 95%),
# then such a difference cannot reasonably be attributed to noise only.
#
# Note that intersecting credible intervals or non-extreme probabilities may
# either indicate there is no discernible difference between the execution time
# of the baseline and optimized files, or that the number of runs is
# insufficient to declare that a statistically significant difference exists.
# It's up to you to interpret which one is the case, and if necessary, run the
# script again increasing the 'num_runs' variable to seek a definitive answer.


# BEST PRACTICES FOR BENCHMARKING
#
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
    num_runs = 30

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
