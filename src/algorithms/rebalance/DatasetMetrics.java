package algorithms.rebalance;

import weka.core.Instance;
import weka.core.Instances;

/**
 * <h1>DatasetMetrics</h1>
 * A simple class for calculating class priors.
 * @author Emmanouil Krasanakis
 */
public class DatasetMetrics {
	/**
	 * <h1>getPriors</h1>
	 * Produces priors that sum to 1 for the given collection of instances.
	 * @param instances the given collection of instances
	 * @return an array of priors
	 */
	public static double[] getPriors(Instances instances) {
		//obtain frequencies
		double[] frequencies = new double[instances.numClasses()];
		for(int i=0;i<instances.numInstances();i++) {
			Instance instance = instances.instance(i);
			frequencies[(int)instance.classValue()]++;
		}
		//convert to probabilities
		double sum = 0;
		for(int i=0;i<frequencies.length;i++)
			sum += frequencies[i];
		if(sum!=0)
			for(int i=0;i<frequencies.length;i++)
				frequencies[i] /= sum;
		return frequencies;
	}
}
