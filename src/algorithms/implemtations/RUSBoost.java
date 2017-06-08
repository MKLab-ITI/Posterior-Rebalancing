package algorithms.implemtations;

import java.io.Serializable;
import java.util.Arrays;

import algorithms.rebalance.DatasetMetrics;
import algorithms.rebalance.DeepCopy;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;

public class RUSBoost extends Classifier implements Serializable {
	private Classifier [] classifiers;
	private double [] classifierWeights;
	private Classifier baseClassifierModel;
	private int T;
	private int Tfinal;
	
	public RUSBoost(Classifier baseClassifierModel, int T) {
		this.baseClassifierModel = baseClassifierModel;
		this.T = T;
	}

	@Override
	public void buildClassifier(Instances instances) throws Exception {
		double[] priors = DatasetMetrics.getPriors(instances);
		
		//initialize list of classifiers and their weights
		classifierWeights = new double[T];
		classifiers = new Classifier[T];
		for(int t=0;t<T;t++)
			classifiers[t] = (Classifier)DeepCopy.copy(baseClassifierModel);
		
		//initialize sample weights
		double[] sampleDistribution = new double[instances.numInstances()];
		for(int i=0;i<sampleDistribution.length;i++)
			sampleDistribution[i] = 1.0/sampleDistribution.length;
		classifierWeights[0] = 1;
		Tfinal = 1;
		double TfinalPerformance = 0;
		
		for(int t=0;t<T;t++) {
			//set distribution as weights
			for(int i=0;i<sampleDistribution.length;i++)
				instances.instance(i).setWeight(sampleDistribution[i]);
			
			//select training set
			weka.filters.supervised.instance.SpreadSubsample res = new weka.filters.supervised.instance.SpreadSubsample();
			res.setInputFormat(instances);
			res.setDistributionSpread(5);//5 is optimal for thoratic
			Instances trainingInstances = Filter.useFilter(instances, res);
			double sumTraining = 0;
			for(int i=0;i<trainingInstances.numInstances();i++)
				sumTraining += trainingInstances.instance(i).weight();
			for(int i=0;i<trainingInstances.numInstances();i++) {
				double prior = priors[(int)trainingInstances.instance(i).classValue()];
				trainingInstances.instance(i).setWeight(trainingInstances.instance(i).weight()/sumTraining*2/prior);
			}
			
			//train classifier
			Classifier classifier = classifiers[t];
			classifier.buildClassifier(trainingInstances);
			
			//calculate classifier errors and estimations
			int prevTFinal = Tfinal;
			Tfinal = t+1;
			int[] truePositives = new int[priors.length];
			int[] positives = new int[priors.length];
			double pseudoLoss = 0;
			double maxPseudoLoss = 0;
			double[] classifierErrors = new double[sampleDistribution.length];
			for(int i=0;i<sampleDistribution.length;i++) {
				int classifierEstimation = (int)classifyInstance(instances.instance(i));
				int classValue = (int)instances.instance(i).classValue();
				double [] distribution = distributionForInstance(instances.instance(i));
				positives[classValue]++;
				if(classifierEstimation==classValue) 
					truePositives[classValue]++;
				else
					classifierErrors[i] = 1-distribution[classValue]+distribution[classifierEstimation];
				pseudoLoss += sampleDistribution[i]*classifierErrors[i]*priors[(int)instances.instance(i).classValue()];
				maxPseudoLoss += 2*sampleDistribution[i]*priors[(int)instances.instance(i).classValue()];
			}
			Tfinal = prevTFinal;
			double performance = 1;
			for(int i=0;i<priors.length;i++)
				performance *= Math.pow((double)truePositives[i]/positives[i], 1.0/priors.length);
			
			if(TfinalPerformance<performance) {
				Tfinal = t+1;
				TfinalPerformance = performance;
			}
			pseudoLoss /= maxPseudoLoss;
			//update distribution
			double updateParameter = pseudoLoss/(1-pseudoLoss);
			for(int i=0;i<sampleDistribution.length;i++)
				sampleDistribution[i] *= Math.pow(updateParameter, 1-0.5*classifierErrors[i]);
			classifierWeights[t] = Math.log(updateParameter);
		
			//normalize distribution
			double sum = 0;
			for(double d : sampleDistribution)
				sum += d;
			for(int i=0;i<sampleDistribution.length;i++)
				sampleDistribution[i] /= sum;
		}
		System.out.println("RUSBoost weights        : "+Arrays.toString(classifierWeights));
		System.out.println("Number of boosts        : "+Tfinal);
		
	}
	
	@Override
	public double[] distributionForInstance(Instance instance) throws Exception {
		double[] ret = classifiers[0].distributionForInstance(instance);
		for(int n=0;n<ret.length;n++)
			ret[n] *= classifierWeights[0];
		for(int t=1;t<Tfinal;t++) {
			double[] tmp = classifiers[t].distributionForInstance(instance);
			for(int n=0;n<ret.length;n++)
				ret[n] += tmp[n]*classifierWeights[t];
		}
		double sum = 0;
		for(double d : ret)
			sum += d;
		for(int n=0;n<ret.length;n++)
			ret[n] /= sum;
		return ret;
	}

	@Override
	public double classifyInstance(Instance instance) throws Exception {
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
