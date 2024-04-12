# Opera

Opera is written in Python using both Poetry and Nix for managing dependencies.
A recent installation of Nix (version 2.18.1 or higher) is the only prerequisite to get started.
Additionally, we offer a Docker-based solution for running Nix.

We support both x86_64 and aarch64 platforms. Note that the binary cache for prebuilt dependencies, including sketch and reduce-algebra, is only available for the x86_64 architecture. On an aarch64 system, building the dependencies might require an additional 10-15 minutes.

The full evaluation of Opera requires sketch, reduce-algebra, and cvc5. Please follow the instructions below to configure your environment.

## Documentation

Below, we provide an overview of the Opera project’s directory structure:
- `b2s/` contains all the sources related to Opera.
    * `cli.py` is the entrypoint that handles command line arguments, initializes converters, and outputs solution.
    * The offline-online program converter of Opera is located at `converters/obs_func_converter.py`.
    * Our IR language definition can be found in `lang_parser.lark`, `lang_parser.py`, and `lang_interp.py`. Our rule-based IR translator is located in `pyast_to_lang.py`.
- `benchmarks/` contains benchmarks for Opera (`stats` and `nexmark`) and baseline evaluations (`sketch` and `sygus`).
- `evaluation/` contains the script for evaluation. The evaluation script `run.sh` will generate figures in PDF format and save raw output files under `evaluation/output`.

## Usage

`nix run .#opera -- --help` lists all the available options for the tool.

- `--syn_unroll_depth` allows the user to set the maximum number of symbolic list elements used in `MineExpressions`.
- `--syn_num_inputs` determines the number of random inputs required in Polynomial Interpolation (refer to Appendix B for details).
- `--syn_ablation` specifies the type of ablation option to run.

Below is an example of how to run Opera on the arithmetic mean benchmark:

```shell
$ nix run .#opera -- benchmarks/stats/mean.py
```

## Getting Started with Nix

Prerequisites:
* Install the [Nix package manager](https://zero-to-nix.com/start/install/).
* If you didn't use the provided link above to install Nix, make sure to enable [Nix flakes](https://nixos.wiki/wiki/Flakes).

To begin, execute the following commands:

```shell
$ nix build .
$ nix develop
```

You can move forward to the section *Step-by-Step Instructions of Opera Evaluation*.

## Getting Started with Nix Container

To start, build the Docker image with the following command:
   
```shell
$ docker build . -t opera_ae:latest
```

Then, create a container from the built image, mapping the current project directory to a path within the container:

```shell
$ docker run --privileged --rm -v $(pwd):/workspace -it opera_ae:latest
```

(Note: The --privileged flag is required due to poetry2nix failing without Nix sandboxing.)

You can move forward to the section *Step-by-Step Instructions of Opera Evaluation*.

## Step-by-Step Instructions of Opera Evaluation

0. Enter the development environment for Opera

```shell
$ nix develop
```

1. Verify the environment

Verify the environment by running the unit tests and testing Opera on a simple example.

```shell
$ pytest
...
=== 85 passed, 2 warnings in 1.58s ===

$ nix run .#opera -- benchmarks/stats/mean.py
...
******************************
  Solution: \prev_out prev_s prev_len x__ -> (let S = {} in let S = S{s = 0} in let S = S{s = prev_s + x__} in (S[s] / (prev_len + 1)), (prev_s + x__, (prev_len + 1)))
******************************
   QE time: 0.163s
  Syn time: 0.091s
Parse time: 0.002s
 Test time: 0.160s
Total time: 0.415s
******************************
...
```

2. Run the evaluation script

We have provided a bash script that runs the experiments described in Section 7 of our paper.
The full evaluation could take up to 3 hours to complete, so we have included an option to run a subset of the experiments. The details for each experiment are presented in the following table:

| Experiment      | Est. Running Time | Outputs                                |
|-----------------|-------------------|----------------------------------------|
| Opera           | 0.5 hour          | `evaluation/output/{stats, nexmark}`   |
| Opera Ablations | 1 hour            | `evaluation/output/{nodecomp, nosymb}` |
| CVC5            | 0.5-2 hours       | `evaluation/output/cvc5`               |
| Sketch          | 0.5-2 hours       | `evaluation/output/sketch`             |


The evaluation script will analyze results from the experiments and print relevant statistics, as discussed in Section 7, to standard output. Use the following commands to run the evaluation script:

```shell
$ cd evaluation

# command to run the full evaluation
$ yes | bash ./run.sh

# select the individual experiments
$ bash ./run.sh

Rerun Opera + ablation evaluation? [y/N]
Rerun CVC5 evaluation? [y/N]
Rerun Sketch evaluation? [y/N]
Run evaluation script? [Y/n]
```

We give an example of running the evaluation script below.

```shell
$ cd evaluation
$ yes | bash ./run.sh

Rerun Opera + ablation evaluation? [y/N] y
Rerun CVC5 evaluation? [y/N] y
Rerun Sketch evaluation? [y/N] y
Run evaluation script? [Y/n] y

LaTeX not found, using default font
Changing working directory to /workspaces/batch2stream


---- Table 1: Statistics about the benchmark set ----
+---------+--------------------------------+-------------------------------+----------------------------------+---------------------------------+
| type    |   ('mean', 'offline_ast_size') |   ('mean', 'online_ast_size') |   ('median', 'offline_ast_size') |   ('median', 'online_ast_size') |
|---------+--------------------------------+-------------------------------+----------------------------------+---------------------------------|
| stats   |                             26 |                            46 |                               24 |                              41 |
| nexmark |                             79 |                            76 |                               42 |                              44 |
+---------+--------------------------------+-------------------------------+----------------------------------+---------------------------------+


---- Table 2: Main synthesis result (Percentage) ----
+---------------------+---------+-----------+
|                     | Stats   | Auction   |
|---------------------+---------+-----------|
| ('Opera', 'count')  | 97%     | 100%      |
| ('CVC5', 'count')   | 30%     | 22%       |
| ('Sketch', 'count') | 12%     | 17%       |
+---------------------+---------+-----------+


---- Figure 11: Comparison between Opera and baselines ----
Saved to evaluation/fig11a-stats-baseline.pdf
Saved to evaluation/fig11b-auction-baseline.pdf


---- Result for RQ2: Baseline comparison ----
Opera outperforms existing SyGuS solvers, synthesizing 3.6× and 7.1× as many tasks as CVC5 and Sketch respectively.


---- Result for RQ1: Opera Result ----
Opera can automatically synthesize 50 out of 51 online schemes with an average synthesis time of 29.4 seconds.


---- Result for RQ3: % of benchmarks solved by ablations ----
Opera: 98.04% (0.00% less than Opera)
Opera-NoSymbolic: 74.51% (23.53% less than Opera)
Opera-NoDecomp: 64.71% (33.33% less than Opera)


---- Figure 13: Comparison between Opera and its ablations ----
Saved to evaluation/fig13-ablation.pdf
```

3. Evaluate the claims presented in the paper

The evaluation script outputs the following result to the standard output:
    1. The statistics discussed in RQ1, RQ2, and RQ3, and
    2. A summary of the experiment result in Table 1 and 2.

Furthermore, the script generates **three** figures in Section 7:

Cumulative distribution functions for
  a. Comparison between Opera and baselines, Stats (Figure 11a) `evaluation/fig11a-stats-baseline.pdf`
  b. Comparison between Opera and baselines, Auction (Figure 11b) `evaluation/fig11b-auction-baseline.pdf`
  c. Comparison between Opera and its ablations (Figure 13) `evaluation/fig13-ablation.pdf`

Note that the final results may vary depending on the performance of the REDUCE solver on the test platform. For reference, we have provided a copy of the evaluation results under `evaluation/output-copy`.
