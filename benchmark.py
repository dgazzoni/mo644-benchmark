#!/usr/bin/env python3

import csv
import engineering_notation
import hashlib
import numpy as np
import os
import platform
import shutil
import scipy.stats
import statistics
import subprocess
import sys
import time
import tqdm
import warnings

## HOW TO USE THIS SCRIPT
#
# Save it in the folder that contains CMakeLists.txt.
#
# Install required packages using pip: engineering_notation, numpy, scipy and
# tqdm.
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


## HOW TO INTERPRET THE OUTPUT
#
# Here is some sample output:
#
#############################################################################
#
# Baseline:
#         Test 1, speedup: min = 0.0270x avg = 0.0308x (execution time: serial avg/stdev = 1.196u/402.378n, parallel avg/stdev = 38.884u/959.367n)
#         Test 2, speedup: min = 2.6004x avg = 2.2335x (execution time: serial avg/stdev = 1.208m/6.775u, parallel avg/stdev = 540.796u/28.729u)
#         Test 3, speedup: min = 3.9857x avg = 3.9763x (execution time: serial avg/stdev = 1.378/9.783m, parallel avg/stdev = 346.583m/2.200m)
#         Test 4, speedup: min = 3.9768x avg = 3.9819x (execution time: serial avg/stdev = 2.724/30.912m, parallel avg/stdev = 684.210m/5.244m)
#         Test 5, speedup: min = 3.9154x avg = 3.9624x (execution time: serial avg/stdev = 5.886/374.699m, parallel avg/stdev = 1.485/89.504m)
#
# File xxx:
#         Test 1, speedup: min = 1.0000x avg = 1.1889x (execution time: serial avg/stdev = 1.196u/402.378n, parallel avg/stdev = 1.006u/77.304n)
#         Baseline CI = (38.800u,38.968u), current test CI = (999.208n,1.013u) p = 5.84e-165 (statistically significant)
#         Test 2, speedup: min = 4.3942x avg = 4.1235x (execution time: serial avg/stdev = 1.208m/6.775u, parallel avg/stdev = 292.920u/12.616u)
#         Baseline CI = (538.272u,543.320u), current test CI = (291.811u,294.029u) p = 5.84e-165 (statistically significant)
#         Test 3, speedup: min = 4.6871x avg = 4.4693x (execution time: serial avg/stdev = 1.378/9.783m, parallel avg/stdev = 308.357m/12.544m)
#         Baseline CI = (346.390m,346.776m), current test CI = (307.255m,309.459m) p = 7.68e-164 (statistically significant)
#         Test 4, speedup: min = 4.6958x avg = 4.5402x (execution time: serial avg/stdev = 2.724/30.912m, parallel avg/stdev = 600.077m/22.037m)
#         Baseline CI = (683.749m,684.671m), current test CI = (598.141m,602.013m) p = 5.84e-165 (statistically significant)
#         Test 5, speedup: min = 5.2074x avg = 5.5184x (execution time: serial avg/stdev = 5.886/374.699m, parallel avg/stdev = 1.067/29.626m)
#         Baseline CI = (1.478,1.493), current test CI = (1.064,1.069) p = 5.84e-165 (statistically significant)
#############################################################################
#
# Speedups are obtained, as expected, by dividing the serial execution time
# by the corresponding parallel execution time (the baseline or one of the
# other files). Both minimum and averages are reported; minimum considers
# the fastest run of the serial and parallel versions for computing the
# speedup. Averages and standard deviations for each serial and parallel
# implementation are also reported.
#
# Confidence intervals (CIs) are reported for the execution times of both
# the baseline and optimized files. In general, if the baseline CI does not
# intersect the optimized CI, the result should be statistically significant
# (i.e. there is an actual difference, not attributable to measurement noise,
# between the baseline and optimized execution times).
# 
# This is also reported in the form of a p-value, taking p < 0.05 as a
# threshold for statistical significance, using the Wilcoxon rank-sum
# non-parametric test.
# 
# Note that the lack of statistical significance may either indicate there is
# no discernible difference between the execution time of the baseline and
# optimized files, or that the number of runs is insufficient to narrow down
# the confidence intervals enough to declare that a statistically significant
# difference exists. It's up to you to interpret which one is the case in your
# runs, and if necessary, run the script again increasing the 'num_runs'
# variable to try to achieve statistical significance.



## BEST PRACTICES FOR BENCHMARKING
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



## DEFINITIONS

# How many repeated runs (of all files and tests)
# Use more if needed to achieve statistical significance
num_runs = 10

# Which tests to run (one for each file in the tests/ folder) 
# Do not add .in extension, it's done automatically
tests = ['1', '2', '3', '4', '5']

# Folder for executable files; may be different under
# Windows when using VS's version of cmake
executable_path = 'build' # os.path.join('build', 'Release')

# Default name for the executable of the parallel code
# In principle this shouldn't be changed
parallel_executable = 'parallel'

# Extension for parallel files, typically either '.c' or '.cu'
parallel_extension = '.c'

# Default name for the source file of the parallel code (note this file will
# be overwritten by the script with the baseline and optimized files)
# This changes from one exercise to the next
parallel_src = 'prime-parallel'

# Name of your baseline file (the one others will be compared against)
baseline_src = 'prime-parallel-baseline'

# List of your optimized files to be compared against the baseline file
files = ['prime-parallel-opt1', 'prime-parallel-opt2']

# Choice of compiler, or None to let CMake choose automatically
c_compiler   = None # 'gcc-10' # 'icc' # 'icl' # 'x86_64-w64-mingw32-gcc-10'
cxx_compiler = None # 'g++-10' # None  # 'icl' # 'x86_64-w64-mingw32-g++'

# Extra C and C++ compiler flags that may be required for some reason
# Use None if no extra flags are necessary
c_flags   = None
cxx_flags = None

# As explained above, this adds a delay between each run of the code to cool
# off the CPU and reduce, or ideally prevent, thermal throttling. This is a
# dict mapping tests (as in, elements of the 'tests' variable above) to the
# number of seconds the CPU will sleep after a given test. The default value
# of None means no delays are inserted
delay_run = None # { '1': 0, '2': 0, '3': 1, '4': 3, '5': 6 }

# Name of a file to save the measurements to (in CSV format), or None
csv_output_file = None



## OS-specific definitions
if platform.system() != 'Windows':
    executable_extension = ''
    generator = ''
else:
    executable_extension = '.exe'
    generator = ''
    # If using Visual Studio, use the generator line below
    # generator = '-G "NMake Makefiles" '



## Miscellaneous functions
def eng(x):
    return engineering_notation.EngNumber(x, precision=3)

def delay(delay_var, delay_name):
    if delay_var > 0:
        if delay_name is not None:
            print('Sleeping between {} for {}s'.format(delay_name, delay_var), flush=True)
        time.sleep(delay_var)

def save_to_csv(filename, num_runs, tests, baseline_src, files, serial_time, parallel_time):
    all_files = [baseline_src] + files
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')

        writer.writerow(['Test', 'Run', 'Serial'] + all_files)

        for test in tests:
            for run in range(num_runs):
                row = [test,run+1,serial_time[test][run]]

                for f in all_files:
                    row += [parallel_time[test][f][run]]

                writer.writerow(row)



## Compilation
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
    os.system('cmake ' + generator + compiler_string + '-DCMAKE_BUILD_TYPE=Release ..')
    os.chdir('..')

def compile_code():
    # Cleaning is required because cmake sometimes doesn't recognize changes
    # after copying a new version of the source file in compile_all().
    subprocess.run(['cmake', '--build', 'build', '--target', 'clean'])
    r = subprocess.run(['cmake', '--build', 'build', '--config', 'Release'])
    if r.returncode != 0:
        sys.stderr.write("Compile error\n")
        sys.exit(1)

def compile_all(executable_path, parallel_executable, executable_extension, parallel_extension, parallel_src, baseline_src, files):
    shutil.copy(os.path.join('src', baseline_src + parallel_extension), os.path.join('src', parallel_src + parallel_extension))
    compile_code()
    shutil.copy(os.path.join(executable_path, parallel_executable + executable_extension), os.path.join(executable_path, baseline_src + executable_extension))

    for f in files:
        shutil.copy(os.path.join('src', f + parallel_extension), os.path.join('src', parallel_src + parallel_extension))
        compile_code()
        shutil.copy(os.path.join(executable_path, parallel_executable + executable_extension), os.path.join(executable_path, f + executable_extension))



## Statistical functions
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m-h, m+h

def calculate_statistics(serial_time, parallel_time):
    min_speedup = min(serial_time)/min(parallel_time)
    avg_serial_time = statistics.mean(serial_time)
    stdev_serial_time = statistics.stdev(serial_time)
    avg_parallel_time = statistics.mean(parallel_time)
    stdev_parallel_time = statistics.stdev(parallel_time)
    avg_speedup = avg_serial_time/avg_parallel_time

    # TODO: maybe turn this into a class? This huge tuple is ugly
    return (min_speedup, avg_serial_time, stdev_serial_time, avg_parallel_time, stdev_parallel_time, avg_speedup)

def statistical_analysis(parallel_baseline, parallel_time):
    lb_baseline, ub_baseline = mean_confidence_interval(parallel_baseline)
    lb, ub = mean_confidence_interval(parallel_time)
    stat, p = scipy.stats.ranksums(parallel_baseline, parallel_time)
    return (lb_baseline, ub_baseline, lb, ub, p)

def compute_and_print_statistics(tests, baseline_src, files, serial_time, parallel_time):
    print('Baseline:')
    for test in tests:
        (min_speedup, avg_serial_time, stdev_serial_time, avg_parallel_time, stdev_parallel_time, avg_speedup) = calculate_statistics(serial_time[test], parallel_time[test][baseline_src])
        print('\tTest {}, speedup: min = {:.4f}x avg = {:.4f}x (execution time: serial avg/stdev = {}/{}, parallel avg/stdev = {}/{})'.format(test, min_speedup, avg_speedup, eng(avg_serial_time), eng(stdev_serial_time), eng(avg_parallel_time), eng(stdev_parallel_time)))

    print()

    for f in files:
        print('File {}:'.format(f))
        for test in tests:
            (min_speedup, avg_serial_time, stdev_serial_time, avg_parallel_time, stdev_parallel_time, avg_speedup) = calculate_statistics(serial_time[test], parallel_time[test][f])
            print('\tTest {}, speedup: min = {:.4f}x avg = {:.4f}x (execution time: serial avg/stdev = {}/{}, parallel avg/stdev = {}/{})'.format(test, min_speedup, avg_speedup, eng(avg_serial_time), eng(stdev_serial_time), eng(avg_parallel_time), eng(stdev_parallel_time)))
            (lb_baseline, ub_baseline, lb, ub, p) = statistical_analysis(parallel_time[test][baseline_src], parallel_time[test][f])
            if p < 0.05:
                msg = 'statistically significant'
            else:
                msg = 'NOT statistically significant'
            print('\tBaseline CI = ({},{}), current test CI = ({},{}) p = {:.3g} ({})'.format(eng(lb_baseline), eng(ub_baseline), eng(lb), eng(ub), p, msg))



## Test runners
def run_one_test(file_path, test_path, pbar = None, pbar_value = None):
    t = subprocess.run([file_path, test_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output_hash = hashlib.sha256(t.stdout).hexdigest()
    exec_time = float(t.stderr)
    # Avoid division by zero if the test is faster than 1 us
    if exec_time < 1e-6:
        exec_time = 1e-6

    return (exec_time, output_hash)

def run_all_tests(num_runs, tests, executable_path, executable_extension, parallel_src, baseline_src, files, delay_run = None, initial_run = True):
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

    if initial_run == True:
        total = len(tests)*(len(all_files) + 1)
    else:
        total = num_runs

    with tqdm.tqdm(total=total,bar_format='{l_bar}{bar}|{elapsed}<{remaining} ') as pbar:
        for _ in range(num_runs):
            for test in tests:
                test_path = os.path.join('tests', test + '.in')
                serial_path = os.path.join(executable_path, 'serial' + executable_extension)

                if delay_run is not None:
                    delay(delay_run[test], None)

                (t, h) = run_one_test(serial_path, test_path)
                if initial_run == True:
                    pbar.update()

                serial_time[test].append(t)
                serial_hash[test] = h

                for f in all_files:
                    parallel_path = os.path.join(executable_path, f + executable_extension)
                    if delay_run is not None:
                        delay(delay_run[test], None)

                    (t, h) = run_one_test(parallel_path, test_path)
                    if initial_run == True:
                        pbar.update()

                    if h != serial_hash[test]:
                        sys.stderr.write('Error in test {}: output of file {} is different from serial code\n'.format(test, f))
                        sys.exit(1)

                    parallel_time[test][f].append(t)

            if initial_run == False:
                pbar.update()

    return (serial_time, parallel_time)



warnings.simplefilter('ignore')
if delay_run == None:
    print('{} total runs: {} repetitions on {} executables for test files {}'.format(num_runs*(1 + len(files))*len(tests), num_runs, 1 + len(files), tests))
else:
    print('{} total runs: {} repetitions on {} executables for test files {}, with {}-{}s delay between runs'.format(num_runs*(1 + len(files))*len(tests), num_runs, 1 + len(files), tests, min(delay_run.values()), max(delay_run.values())))

print('Generating Makefiles')
generate_makefile(generator, c_compiler, cxx_compiler)

print('Compiling code')
compile_all(executable_path, parallel_executable, executable_extension, parallel_extension, parallel_src, baseline_src, files)

print('Starting initial run')
(st, pt) = run_all_tests(1, tests, executable_path, executable_extension, parallel_src, baseline_src, files, delay_run, initial_run = True)
print('Starting main run')
(serial_time, parallel_time) = run_all_tests(num_runs, tests, executable_path, executable_extension, parallel_src, baseline_src, files, delay_run, initial_run = False)

compute_and_print_statistics(tests, baseline_src, files, serial_time, parallel_time)

if csv_output_file is not None:
    save_to_csv(csv_output_file, num_runs, tests, baseline_src, files, serial_time, parallel_time)



## TODOs:
# TODO: less runs if a test takes longer, more runs if it's shorter? Would that be enough for statistical significance?
# TODO: run some high-performance code for 10-50ms before running tests, to "spool up" TurboBoost
# TODO: select number of runs automatically to achieve statistical signifance
# TODO: select delay between runs automatically (how?)
# TODO: estimate confidence interval for speedup (possibly using bootstrap methods) for comparison with the TAs' speedups
# TODO: replace statistical tests by Bayesian tests
# TODO: test compiler selection code on Windows
# TODO: measure actual execution time of each test run, and use this to update the progress bar more frequently (but still somewhat accurately)
