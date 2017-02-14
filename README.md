# Posterior-Rebalancing
This repository contains the experimental testbed for the paper **Posterior Rebalance as a Method to Reduce Classifier Bias on Imbalanced Problems**.

## Main Idea
Our algorithm aims to "repair" classification posteriors by giving more weight to posteriors of minority classes.

## Using our Classifier
Our proposed posterior rebalancing method is available as a Weka classifier in *src/algorithms/rebalance/ClassRebalance* (for examples of how to select parameters for this classifier, see *src/algorithms/rebalance/ExperimentScheme*).

## Experiment Results
Experiment results are available in the *out/* and a partial summary of those used in our paper is presented in *Online Appendix.pdf*
