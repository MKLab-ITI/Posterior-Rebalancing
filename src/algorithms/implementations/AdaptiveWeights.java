package algorithms.implementations;

import java.util.Arrays;

import algorithms.rebalance.ClassRebalance;
import algorithms.rebalance.DatasetMetrics;
import algorithms.rebalance.DeepCopy;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;

public class AdaptiveWeights extends Classifier {
	private static final long serialVersionUID = 8309691363092658460L;

	private Classifier baseClassifier;
	private double[] weights;
	private boolean pretrained;
	
	public AdaptiveWeights(Classifier baseClassifier, boolean pretrained) {
		this.baseClassifier = baseClassifier;
		this.pretrained = pretrained;
	}
	
	private void normalizeWeights() {
		double sum = 0;
		for(int i=0;i<weights.length;i++)
			sum += weights[i];
		for(int i=0;i<weights.length;i++)
			weights[i] /= sum;
	}
	
	@Override
	public final void buildClassifier(Instances instances) throws Exception {
		if(!pretrained)
			baseClassifier.buildClassifier(instances);
		double[] priors = DatasetMetrics.getPriors(instances);
		weights = new double[priors.length];
		for(int i=0;i<weights.length;i++)
			weights[i] = 1/priors[i];
		normalizeWeights();
		
		for(int repeat=0;repeat<100;repeat++) {
			System.out.println("Weights: "+Arrays.toString(weights));
			double[] offsets = new double[weights.length];
			for(int i=0;i<instances.numInstances();i++) {
				Instance instance = instances.instance(i);
				int estimationValue = (int)classifyInstance(instance);
				if(estimationValue!=instance.classValue()) 
					offsets[(int) instance.classValue()] += 1;//distributionForInstance(instance)[(int) estimationValue];
			}
			for(int i=0;i<weights.length;i++)
				weights[i] += 1.0/instances.numInstances()/(1+offsets[i])/priors[i];
			normalizeWeights();
		}
	}
	
	@Override
	public final double[] distributionForInstance(Instance instance) throws Exception {
		double[] ret = baseClassifier.distributionForInstance(instance);
		for(int n=0;n<ret.length;n++)
			ret[n] *= weights[n];
		double sum = 0;
		for(double d : ret)
			sum += d;
		for(int n=0;n<ret.length;n++)
			ret[n] /= sum;
		return ret;
	}

	@Override
	public final double classifyInstance(Instance instance) throws Exception {
		double maxClassification = Double.NEGATIVE_INFINITY;
		int selection = -1;
		double[] distribution = distributionForInstance(instance);
		for(int i=0;i<distribution.length;i++) 
			if(maxClassification<=distribution[i]) {
				maxClassification = distribution[i];
				selection = i;
			}
		return selection;
	}
}
