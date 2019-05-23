# Description

Use Ian Vernon's protocol design criterion.
Basically do global sensitivity, randomly picking from model parameter prior and forward simulating from a given protocol 1000s of times, then loop over and give a score based on how 'different' protocol output is to ones from other model parameter samples.
Sort of a global sensitivity via brute force.

- `prior-parameters`: prior model parameters; e.g. parameters obtained via fitting to different protocols with different output.
