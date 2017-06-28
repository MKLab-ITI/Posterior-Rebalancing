package algorithms.implementations;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

import algorithms.rebalance.DatasetMetrics;
import algorithms.rebalance.DeepCopy;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;

public abstract class Boosting extends Classifier implements Serializable {
	private static final long serialVersionUID = -8058566690484220883L;
	private Classifier [] classifiers;
	private double [] classifierWeights;
	private Classifier baseClassifierModel;
	private int Tfinal;
	
	public Boosting(Classifier baseClassifierModel) {
		this.baseClassifierModel = baseClassifierModel;
	}
	
	public abstract Instances generateBoostDataset(Instances instances, int classifierId, int numberOfClassifiers) throws Exception;
	public abstract boolean keepOnlyFavorableBoosts();
	public abstract int calculateNumberOfClassifiers(Instances instances) throws Exception;
	
	@Override
	public final void buildClassifier(Instances instances) throws Exception {
		double[] priors = DatasetMetrics.getPriors(instances);
		int T = calculateNumberOfClassifiers(instances);
		
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
				int classifierEstimation = (int)classifyInstance(instances.instance(i));
				int classValue = (int)instances.instance(i).classValue();
				double [] distribution = distributionForInstance(instances.instance(i));
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
			
			if(TfinalPerformance<performance || !keepOnlyFavorableBoosts()) {
				Tfinal = t+1;
				TfinalPerformance = performance;
			}
			{
				pseudoLoss /= maxPseudoLoss;
				//System.out.println("loss "+pseudoLoss);
				//update distribution
				double updateParameter = pseudoLoss/(1-pseudoLoss);
				for(int i=0;i<sampleDistribution.length;i++)
					sampleDistribution[i] *= Math.pow(updateParameter, 1-0.5*classifierErrors[i]);
				classifierWeights[t] = -Math.log(updateParameter);
			}
		
			//normalize distribution
			double sum = 0;
			for(double d : sampleDistribution)
				sum += d;
			for(int i=0;i<sampleDistribution.length;i++)
				sampleDistribution[i] /= sum;
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
	
	
	
	public static class RUSBoost extends Boosting {
		private static final long serialVersionUID = 5091693448499798955L;
		private int T;
		public RUSBoost(Classifier baseClassifierModel, int T) {
			super(baseClassifierModel);
			this.T = T;
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
	
	
	
	public static class ClusterBoost extends Boosting {
		private static final long serialVersionUID = 2222702274438426399L;

		public ClusterBoost(Classifier baseClassifierModel) {
			super(baseClassifierModel);
		}

		@Override
		public int calculateNumberOfClassifiers(Instances instances) throws Exception {
			if(clusters.isEmpty()) 
				clusters = buildClusters(instances);
			return clusters.size();
		}
		
		protected ArrayList<ArrayList<Instance>> buildClusters(Instances instances) throws Exception {
			double[] priors = DatasetMetrics.getPriors(instances);
			if(priors.length!=2)
				throw new Exception("Clustered boosting can only be performed on binary problems");
			int majorityClass = priors[0]<priors[1]?1:0;
			int numberOfClusters = (priors[0]<priors[1])?(int)Math.ceil(priors[1]/priors[0]):(int)Math.ceil(priors[0]/priors[1]);
			ArrayList<Instance> majorityInstances = new ArrayList<Instance>();
			for(int i=0;i<instances.numInstances();i++)
				if(instances.instance(i).classValue()==majorityClass)
					majorityInstances.add(instances.instance(i));
			System.out.println("Discovered "+majorityInstances.size()+" majority class instances to split into "+numberOfClusters+" clusters");
			
			HashMap<double[], ArrayList<Instance>> clusters = new HashMap<double[], ArrayList<Instance>>();
			HashMap<Instance, double[]> instanceClusters = new HashMap<Instance, double[]>();
			int classIndex = instances.classIndex();
			for(int cl=0;cl<numberOfClusters;cl++) {
				double[] center = new double[instances.numAttributes()-1];
				for(int i=0;i<classIndex;i++)
					center[i] = Math.random()*instances.meanOrMode(i)*2;
				for(int i=classIndex;i<center.length;i++)
					center[i] = Math.random()*instances.meanOrMode(i+1)*2;
				clusters.put(center, new ArrayList<Instance>());
				//System.out.println("Center: "+Arrays.toString(center));
			}
			System.out.print("Progress ");
			
			int repeat = majorityInstances.size()+1;
			while(repeat!=0) {
				repeat = 0;
				//reclassify instance
				for(Instance instance : majorityInstances) {
					double[] prevCenter = instanceClusters.get(instance);
					double[] nearestCenter = null;
					double nearestDistance = Double.POSITIVE_INFINITY;
					for(double[] center : clusters.keySet()) {
						double distance = 0;
						for(int i=0;i<classIndex;i++)
							if(!Double.isNaN(instance.value(i)))
								distance += (center[i]-instance.value(i))*(center[i]-instance.value(i));
						for(int i=classIndex;i<center.length;i++)
							if(!Double.isNaN(instance.value(i+1)))
								distance += (center[i]-instance.value(i+1))*(center[i]-instance.value(i+1));
						if(distance<nearestDistance) {
							nearestDistance = distance;
							nearestCenter = center;
						}
					}
					
					//change cluster for instance if necessary
					if(prevCenter!=nearestCenter) {
						if(prevCenter!=null)
							clusters.get(prevCenter).remove(instance);
						clusters.get(nearestCenter).add(instance);
						instanceClusters.put(instance, nearestCenter);
						repeat++;
					}
				}
				
				System.out.print("-");
				//System.out.println("-------------------");
				for(double[] center : new ArrayList<double[]>(clusters.keySet())) 
					if(clusters.get(center).isEmpty()){
						//clusters.remove(center);//instead of removing create new center
						int count = 0;
						for(double[] otherCenter : clusters.keySet()) 
							if(!clusters.get(otherCenter).isEmpty()) {
								for(int i=0;i<center.length;i++)
									center[i] += otherCenter[i];
								count++;
							}
						for(int i=0;i<center.length;i++)
							center[i] *= Math.random()*2/count;
					}
				//recalculate centers
				for(double[] center : clusters.keySet()) {
					for(int i=0;i<center.length;i++)
						center[i] = 0;
					for(Instance instance : clusters.get(center)) {
						for(int i=0;i<classIndex;i++)
							if(!Double.isNaN(instance.value(i)))
								center[i] += instance.value(i);
						for(int i=classIndex;i<center.length;i++)
							if(!Double.isNaN(instance.value(i+1)))
								center[i] += instance.value(i+1);
					}
					if(clusters.get(center).size()!=0)
						for(int i=0;i<center.length;i++)
							center[i] /= clusters.get(center).size();
					//System.out.println("Center ("+clusters.get(center).size()+")\t: "+Arrays.toString(center));
				}
			}

			for(double[] center : new ArrayList<double[]>(clusters.keySet())) 
				if(clusters.get(center).isEmpty())
					clusters.remove(center);
			
			System.out.println("\nFinal split into "+clusters.size()+" clusters");
			
			return new ArrayList<ArrayList<Instance>>(clusters.values());
		}
		
		private ArrayList<ArrayList<Instance>> clusters = new ArrayList<ArrayList<Instance>>();

		@Override
		public Instances generateBoostDataset(Instances instances, int classifierId, int numberOfClassifiers) throws Exception {
			ArrayList<Instance> cluster = clusters.get(classifierId);
			Instances trainingInstances = new Instances(instances, cluster.size());//initialize empty set of instances
			for(Instance instance : cluster) {
				Instance temp = new Instance(instance);
				temp.setDataset(trainingInstances);
				trainingInstances.add(temp);
			}
			return trainingInstances;
		}

		@Override
		public boolean keepOnlyFavorableBoosts() {
			return true;
		}
		
	}
}
