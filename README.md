# Tensorize: Fast Synthesis of Tensor Programs from Legacy Code using Symbolic Tracing, Sketching and Solving

Tensor domain specific languages (DSLs) achieve substantial performance due to high-level compiler optimization and hardware acceleration.
However, to achieve such performance for existing applications requires the  programmer to manual rewrite their legacy code in evolving Tensor DSLs.
Prior efforts to automate this translation face significant scalability issues which greatly reduces their applicability to real-world code.

This paper presents Tensorize, a novel MLIR-based compiler approach to automatically lift legacy code to high level Tensor DSLs using program synthesis.
Tensorize uses a symbolic trace of the legacy program as a specification and automatically selects sketches from the target Tensor DSLs to drive the program synthesis. 
It uses an algebraic solver to rapidly simplify the specification, resulting in a fast, automatic approach that is correct by design.

We evaluate Tensorize on  several legacy code benchmarks and compare against state-of-the-art techniques. Tensorize is able to lift more code than prior schemes, is an order of magnitude faster in synthesis time, and guarantees correctness by construction.

## Link to Paper

[https://dl.acm.org/doi/10.1145/3696443.3708956](https://dl.acm.org/doi/10.1145/3696443.3708956)

## Reproducing Paper Results

Run our docker-based workflow with
```
./ run_all . sh
```

## Development Setup

Use the following scripts to first build the dependencies and then build the Tensorize project.

```
./build_tools/build_dependencies.sh
./build_tools/build.sh
```
