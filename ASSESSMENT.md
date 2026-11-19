# Original self-study assessment

Not an official NUS rubric.

## Which components actually train?

The XOR network trains both layers with backpropagation. Fixed convolution, scalar recurrence and seeded attention are mechanism demonstrations.

## Why divide QK transpose by square root of key dimension?

It controls score scale as dimensionality increases, reducing extreme softmax saturation.

## Build task

Implement a framework CNN on a small public image dataset, compare with a non-neural baseline, and log seeds, curves and errors. Do not call the fixed filter a trained CNN.

## Suggested scoring, 100 points
- Reproducible clean run and tests: 25
- Correct mechanism and explanation: 25
- Evaluation with a held-out set or explicit mechanism checks: 20
- Failure cases, provenance and limits: 20
- Readable demo and documentation: 10

Pass suggestion: 70, with no fabricated results, private-data exposure or unsupported production claims. This is a self-study convention.
