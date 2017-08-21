package algorithms.rebalance;

import java.io.Serializable;
import java.util.Arrays;

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;

/**
 * <h1>Boosting</h1>
 * An abstract class which presents basic utility for AdaBoost ensembles of a resampled base classifier.
 * @author Emmanouil Krasanakis
 */
public abstract class Boosting extends Classifier implements Serializable {
	private static final long serialVersionUID = -8058566690484220883L;
	private Classifier [] classifiers;
	private double [] classifierWeights;
	private int Tfinal;
	
	protected Boosting() {
	}
	
	public abstract Instances generateBoostDataset(Instances instances, int classifierId, int numberOfClassifiers) throws Exception;
	public abstract boolean keepOnlyFavorableBoosts();
	public abstract int calculateNumberOfClassifiers(Instances instances) throws Exception;
	public abstract Classifier produceClassifier(int classifierId, int numberOfClassifiers) throws Exception;
	
	@Override
	public final void buildClassifier(Instances instances) throws Exception {
		double[] priors = DatasetMetrics.getPriors(instances);
		int T = calculateNumberOfClassifiers(instances);
		
		//initialize list of classifiers and their weights
		classifierWeights = new double[T];
		classifiers = new Classifier[T];
		for(int t=0;t<T;t++)
			classifiers[t] = produceClassifier(t,T);
		
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
			Instances trainingInstances = generateBoostDataset(instances, t, T);
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
			
			for(int i=0;i<classifierErrors.length;i++) {
				int classifierEstimation = (int)classifier.classifyInstance(instances.instance(i));
				double [] distribution = classifier.distributionForInstance(instances.instance(i));
				int classValue = (int)instances.instance(i).classValue();
				positives[classValue]++;
				if(classifierEstimation==classValue) 
					truePositives[classValue]++;
				else
					classifierErrors[i] = 1-distribution[classValue]+distribution[classifierEstimation];
				pseudoLoss += sampleDistribution[i]*classifierErrors[i]*priors[classValue];
				maxPseudoLoss += 2*sampleDistribution[i]*priors[classValue];
			}
			
			Tfinal = prevTFinal;
			double performance = 1;
			for(int i=0;i<priors.length;i++)
				performance *= Math.pow((double)truePositives[i]/positives[i], 1.0/priors.length);
			//System.out.println("t="+t+" "+performance);
			if(performance==0)
				break;
			
			if(TfinalPerformance<performance || !keepOnlyFavorableBoosts()) {
				Tfinal = t+1;
				TfinalPerformance = performance;
			}
			{
				pseudoLoss /= maxPseudoLoss;
				if(pseudoLoss==0) {
					for(int i=0;i<classifierWeights.length;i++)
						classifierWeights[i] = 0;
					classifierWeights[t] = 1;
					break;
				}
				//System.out.println("loss "+pseudoLoss);
				//update distribution
				double updateParameter = pseudoLoss/(1-pseudoLoss);
				for(int i=0;i<sampleDistribution.length;i++)
					sampleDistribution[i] *= Math.pow(updateParameter, 1-0.5*classifierErrors[i]);
				classifierWeights[t] = -Math.log(updateParameter)*performance;
			}
		}
		System.out.println("Number of boosts        : "+Tfinal);
		System.out.println("Boosting weights        : "+Arrays.toString(classifierWeights));
		
	}
	
	@Override
	public final double[] distributionForInstance(Instance instance) throws Exception {
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
	
	
	public static class RebalanceBoost extends Boosting {
		private static final long serialVersionUID = 5091693448499798955L;
		private int T;
		private Classifier baseClassifierModel;
		public RebalanceBoost(Classifier baseClassifierModel, int T) {
			this.baseClassifierModel = baseClassifierModel;
			this.T = T;
		}
		public Classifier produceClassifier(int classifierId, int numberOfClassifiers) throws Exception {
			return new ClassRebalance((Classifier)DeepCopy.copy(baseClassifierModel), weka.core.Utils.splitOptions("-function lin -rebalance "+2*(double)classifierId/numberOfClassifiers));
		}
		@Override
		public int calculateNumberOfClassifiers(Instances instances) {
			return T;
		}
		@Override
		public Instances generateBoostDataset(Instances instances, int classifierId, int numberOfClassifiers) throws Exception {
			weka.filters.supervised.instance.SpreadSubsample res = new weka.filters.supervised.instance.SpreadSubsample();
			res.setInputFormat(instances);
			res.setDistributionSpread(5);//5 is optimal for thoratic
			return Filter.useFilter(instances, res);
		}
		@Override
		public boolean keepOnlyFavorableBoosts() {
			return false;
		}
	}
	
	/**
	 * <h1>RUSBoost</h1>
	 * Implements the RUSBoost method over a given base classifier.
	 * @author Emmanouil Krasanakis
	 */
	public static class RUSBoost extends Boosting {
		private static final long serialVersionUID = 5091693448499798955L;
		private int T;
		private Classifier baseClassifierModel;
		public RUSBoost(Classifier baseClassifierModel, int T) {
			this.baseClassifierModel = baseClassifierModel;
			this.T = T;
		}
		public Classifier produceClassifier(int classifierId, int numberOfClassifiers) throws Exception {
			return (Classifier)DeepCopy.copy(baseClassifierModel);
		}
		@Override
		public int calculateNumberOfClassifiers(Instances instances) {
			return T;
		}
		@Override
		public Instances generateBoostDataset(Instances instances, int classifierId, int numberOfClassifiers) throws Exception {
			weka.filters.supervised.instance.SpreadSubsample res = new weka.filters.supervised.instance.SpreadSubsample();
			res.setInputFormat(instances);
			res.setDistributionSpread(5);//5 is optimal for thoratic
			return Filter.useFilter(instances, res);
		}
		@Override
		public boolean keepOnlyFavorableBoosts() {
			return true;
		}
	}
}
