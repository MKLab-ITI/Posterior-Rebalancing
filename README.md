# Posterior-Rebalancing
This repository contains the experimental testbed for the paper <b>Posterior Rebalance as a Method to Reduce Classifier Bias on Imbalanced Problems</b>.

## Main Idea
Our algorithm aims to "repair" classification posteriors by giving more weight to posteriors of minority classes.

## Using our Classifier
Our proposed posterior rebalancing method is available as a Weka classifier in <it>src/algorithms/rebalance/ClassRebalance</it> (for examples of how to select parameters for this classifier, see <it>src/algorithms/rebalance/ExperimentScheme</it>).
