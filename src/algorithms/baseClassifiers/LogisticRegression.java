package algorithms.baseClassifiers;

import java.io.Serializable;
import java.util.Enumeration;

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

/**
 * <h1>LogisticRegression</h1>
 * This weka {@link Classifier} implements logistic regression. As Weka's logistic regression can produce
 * a lot of features when converting nominal to binary attributes, this classifier can be used to prevent
 * heap overflow from occurring by not employing that process.
 * 
 * @author Krasanakis Emmanouil
 * @deprecated This classifier performs sub-optimally, as it does not implement regularization.
 */
public class LogisticRegression extends Classifier implements Serializable { 
	private static final long serialVersionUID = 6383329433068885155L;
	private double[][] weights;
	private int ITERATIONS = 3000;
	private double learningRate =  0.0001;
	
	@Override
	public void buildClassifier(Instances instances) throws Exception {
		int numClasses = instances.classAttribute().numValues();
		int classIndex = instances.classIndex();
		int numAttributes = instances.numAttributes();
		weights = new double[numClasses][numAttributes-1];
		learningRate =  0.01/instances.numInstances();
		//double err = 0;
		//int num = 0;
		for (int n=0; n<ITERATIONS; n++) {
			@SuppressWarnings("unchecked")
			Enumeration<Instance> enumeration = instances.enumerateInstances();
			while(enumeration.hasMoreElements()) {
				Instance instance = enumeration.nextElement();
				int instanceClass =(int)instance.classValue();
				double predictedClass[] = distributionForInstance(instance);
				for(int cl=0;cl<numClasses;cl++) {
					double label = (cl==instanceClass)?1:0;
					double predictedLabel = predictedClass[cl];
					for (int i=0;i<classIndex;i++)
						if(!instance.isMissing(i))
							weights[cl][i] += learningRate * (label - predictedLabel) * predictedLabel *  instance.value(i);
					for (int i=classIndex+1;i<numAttributes;i++)
						if(!instance.isMissing(i))
							weights[cl][i-1] += learningRate * (label - predictedLabel) * predictedLabel * instance.value(i);
					//err += Math.abs(label-predictedLabel);
					//num++;
				}
			}
			//System.out.println(err/num);
		}
	}

	@Override
	public double[] distributionForInstance(Instance instance) throws Exception {
		double[] ret = new double [weights.length];
		for(int cl=0;cl<ret.length;cl++) {
			double logit = 0;
			for(int i=0;i<instance.classIndex();i++)
				if(!instance.isMissing(i))
					logit += weights[cl][i] * instance.value(i);
			for (int i=instance.classIndex()+1;i<instance.numAttributes();i++)
				if(!instance.isMissing(i))
					logit += weights[cl][i-1] * instance.value(i);
			//System.out.print(logit+" ");
			ret[cl] = 1.0 / (1.0 + Math.exp(-logit));
		}
		//System.out.println(": desired "+instance.classValue());
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
