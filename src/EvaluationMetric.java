import java.util.ArrayList;

import org.apache.commons.math3.stat.inference.MannWhitneyUTest;
import org.apache.commons.math3.stat.inference.TTest;

import algorithms.rebalance.ClassRebalance;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;

/**
 * <h1>EvaluationMetric</h1>
 * This is a common interface between custom (e.g. entropy-aware) imbalance metrics and metrics obtained through Weka {@link Evaluation}.
 * For each cross-validation fold, {@link #validateInstances(Instances, Classifier, Classifier)} should be called. Afterwards,
 * {@link #getValue(Evaluation)} obtains the desired metric value while also taking into account evaluation performed by Weka
 * (certain metrics don't take this information into account, whereas others use only the results of weka's evaluation).
 * <br/>
 * Custom imbalance metrics compare classifiers to their provided base classifier, to examine the effects of mitigating imbalance.
 * @author Emmanouil Krasanakis
 */
public interface EvaluationMetric {
	public default void validateInstances(Instances test, Classifier classifier, Classifier baseClassifier) {
		try{
			for(int i=0;i<test.numInstances();i++) 
				validateInstance(test.instance(i), classifier, baseClassifier);
		}
		catch(Exception e) {
			e.printStackTrace();
		}
	}
	public default void validateInstance(Instance instance, Classifier classifier, Classifier baseClassifier) throws Exception {
		double[] classifierDistribution = classifier.distributionForInstance(instance);
		double[] baseClassifierDistribution = baseClassifier.distributionForInstance(instance);
		int classifierClass = (int)classifier.classifyInstance(instance);
		int baseClassifierClass = (int)baseClassifier.classifyInstance(instance);
		int actualClass = (int)instance.classValue();
		double classifierEntropy = ClassRebalance.normalizedEntropy(classifierDistribution);
		double baseClassifierEntropy = ClassRebalance.normalizedEntropy(baseClassifierDistribution);
		validateInstance(classifierClass, baseClassifierClass, actualClass, classifierEntropy, baseClassifierEntropy);
	}
	public void validateInstance(int classifierClass, int baseClassifierClass, int actualClass, double classifierEntropy, double baseClassifierEntropy);
	public double getValue(Evaluation evaluation);
	public String unit();
	public void clear();

	/**
	 * <h1>toPrimitive</h1>
	 * @param array an array list of doubles
	 * @return an array representation of the give array list
	 */
	public static double[] toPrimitive(ArrayList<Double> array) {
		  if (array == null) {
		    return null;
		  } else if (array.size() == 0) {
		    return new double[0];
		  }
		  final double[] result = new double[array.size()];
		  for (int i = 0; i < array.size(); i++) {
		    result[i] = array.get(i).doubleValue();
		  }
		  return result;
	}
	
	/**
	 * <h1>ILoss</h1>
	 * Novel metric which computes the average information loss in comparison to a base 
	 * classifier when the base classifier samples are accurately computed
	 * (information loss is not considered to occur otherwise).
	 * @author Emmanouil Krasanakis
	 */
	public static class ILoss implements EvaluationMetric {
		private double gain = 0;
		private int n = 0;
		@Override
		public void validateInstance(int classifierClass, int baseClassifierClass, int actualClass, double classifierEntropy, double baseClassifierEntropy) {
			double entropyGain = classifierEntropy - baseClassifierEntropy;
			if(classifierClass==actualClass && baseClassifierClass==actualClass)
				gain += entropyGain;
			else if(classifierClass==actualClass && baseClassifierClass!=actualClass)
				{}//gain -= entropyGain;
			else if(classifierClass!=actualClass && baseClassifierClass==actualClass)
				gain += entropyGain;
			else if(classifierClass!=actualClass && baseClassifierClass!=actualClass)
				{}//gain -= entropyGain;
			n++;
		}
		
		@Override
		public double getValue(Evaluation evaluation) {
			return gain / n;
		}
		
		@Override
		public String unit() {
			return "b";
		}
		
		@Override
		public void clear() {
			gain = 0;
			n = 0;
		}
	}
	/**
	 * <h1>CorrectEntropyTTest</h1>
	 * Performs T-test to check that distribution entropies are similar between the base and the imbalance-aware classifier.
	 * @author Emmanouil Krasanakis
	 */
	public static class CorrectEntropyTTest implements EvaluationMetric {
		private ArrayList<Double> entropiesBase = new ArrayList<Double>();
		private ArrayList<Double> entropies = new ArrayList<Double>();
		@Override
		public void validateInstance(int classifierClass, int baseClassifierClass, int actualClass, double classifierEntropy, double baseClassifierEntropy) {
			if(baseClassifierClass==actualClass)
				entropiesBase.add(baseClassifierEntropy);
			else
				entropiesBase.add(-baseClassifierEntropy);
			if(classifierClass==actualClass)
				entropies.add(classifierEntropy);
			else
				entropies.add(-classifierEntropy);
		}
		@Override
		public double getValue(Evaluation evaluation) {
			TTest tTest = new TTest();
			double T = (double)tTest.tTest(toPrimitive(entropies), toPrimitive(entropiesBase));
			return T;
		}
		@Override
		public String unit() {
			return "T";
		}
		@Override
		public void clear() {
			entropiesBase.clear();
			entropies.clear();
		}
	}

	/**
	 * <h1>CorrectEntropyUTest</h1>
	 * Performs two-tailed U-test to check whether entropy gain for correct classifications is equal to entropy gain 
	 * for incorrect classifications of the imbalance-aware classifier.
	 * @author Emmanouil Krasanakis
	 * @see CorrectEntropyBiserialCorrelation
	 */
	public static class CorrectEntropyUTest implements EvaluationMetric {
		private ArrayList<Double> ranksCorrect = new ArrayList<Double>();
		private ArrayList<Double> ranksIncorrect = new ArrayList<Double>();
		@Override
		public void validateInstance(int classifierClass, int baseClassifierClass, int actualClass, double classifierEntropy, double baseClassifierEntropy) {
			double entropyGain = classifierEntropy - baseClassifierEntropy;
			if(classifierClass==actualClass)
				ranksCorrect.add(-entropyGain);
			else
				ranksIncorrect.add(-entropyGain);
		}
		
		@Override
		public double getValue(Evaluation evaluation) {
			MannWhitneyUTest uTest = new MannWhitneyUTest();
			double U = (double)uTest.mannWhitneyU(toPrimitive(ranksCorrect), toPrimitive(ranksIncorrect));
			U = U/ranksCorrect.size()/ranksIncorrect.size();
			return U;//measures whether distributions differ between correct and incorrect classifications (0.5=don't differ, 1=certain)
		}
		@Override
		public String unit() {
			return "U";
		}
		@Override
		public void clear() {
			ranksCorrect.clear();
			ranksIncorrect.clear();
		}
	}
	/**
	 * <h1>CorrectEntropyBiserialCorrelation</h1>
	 * Calculates the biserial metric for {@link CorrectEntropyUTest}.
	 * @author Emmanouil Krasanakis
	 */
	public static class CorrectEntropyBiserialCorrelation extends CorrectEntropyUTest {
		@Override
		public double getValue(Evaluation evaluation) {
			double U = super.getValue(evaluation);
			return 2*U-1;//rank-biserial correlation, i.e. the proportion of samples that confirm the hypothesis that distributions differ between correct and incorrect
		}
		@Override
		public String unit() {
			return "r";
		}
	}

	/**
	 * <h1>GM</h1>
	 * Calculates the geometric mean between TPRs obtained from Weka's {@link Evaluation}.
	 * @author Emmanouil Krasanakis
	 */
	public static class GM implements EvaluationMetric {
		@Override
		public void validateInstances(Instances test, Classifier classifier, Classifier baseClassifier) {
		}
		@Override
		public void validateInstance(Instance instance, Classifier classifier, Classifier baseClassifier) throws Exception {
		}
		@Override
		public void validateInstance(int classifierClass, int baseClassifierClass, int actualClass, double classifierEntropy, double baseClassifierEntropy) {
		}
		@Override
		public String unit() {
			return "";
		}
		@Override
		public void clear() {
		}
		@Override
		public double getValue(Evaluation evaluation) {
		    double[] classFrequencies = evaluation.getClassPriors();
			double GTPr = 1;
			int count = 0;
		    for(int i=0;i<classFrequencies.length;i++)  {
		    	if(classFrequencies[i]==0)
		    		continue;
			    GTPr *= evaluation.truePositiveRate(i);
			    count++;
		    }
		    GTPr =  Math.pow(GTPr,1.0/count);
		    return GTPr;
		}
	}
	

	/**
	 * <h1>AM</h1>
	 * Calculates the arithmetic mean between TPRs obtained from Weka's {@link Evaluation}.
	 * @author Emmanouil Krasanakis
	 */
	public static class AM implements EvaluationMetric {
		@Override
		public void validateInstances(Instances test, Classifier classifier, Classifier baseClassifier) {
		}
		@Override
		public void validateInstance(Instance instance, Classifier classifier, Classifier baseClassifier) throws Exception {
		}
		@Override
		public void validateInstance(int classifierClass, int baseClassifierClass, int actualClass, double classifierEntropy, double baseClassifierEntropy) {
		}
		@Override
		public String unit() {
			return "";
		}
		@Override
		public void clear() {
		}
		@Override
		public double getValue(Evaluation evaluation) {
		    double[] classFrequencies = evaluation.getClassPriors();
			double meanTPr = 0;
			int count = 0;
		    for(int i=0;i<classFrequencies.length;i++)  {
		    	if(classFrequencies[i]==0)
		    		continue;
			    meanTPr += evaluation.truePositiveRate(i);
			    count++;
		    }
		    meanTPr /= count;
		    return meanTPr;
		}
	}

	/**
	 * <h1>AUC</h1>
	 * Reports the weighted AUC from Weka's {@link Evaluation}.
	 * @author Emmanouil Krasanakis
	 */
	public static class AUC implements EvaluationMetric {
		@Override
		public void validateInstances(Instances test, Classifier classifier, Classifier baseClassifier) {
		}
		@Override
		public void validateInstance(Instance instance, Classifier classifier, Classifier baseClassifier) throws Exception {
		}
		@Override
		public void validateInstance(int classifierClass, int baseClassifierClass, int actualClass, double classifierEntropy, double baseClassifierEntropy) {
		}
		@Override
		public String unit() {
			return "";
		}
		@Override
		public void clear() {
		}
		@Override
		public double getValue(Evaluation evaluation) {
		    return evaluation.weightedAreaUnderROC();
		}
	}

	/**
	 * <h1>Imbalance</h1>
	 * Calculates weighted average differences between TPRs obtained from Weka's {@link Evaluation}.
	 * (For two-class problems, this weighted averages become simple averages.)
	 * @author Emmanouil Krasanakis
	 */
	public static class Imbalance implements EvaluationMetric {
		@Override
		public void validateInstances(Instances test, Classifier classifier, Classifier baseClassifier) {
		}
		@Override
		public void validateInstance(Instance instance, Classifier classifier, Classifier baseClassifier) throws Exception {
		}
		@Override
		public void validateInstance(int classifierClass, int baseClassifierClass, int actualClass, double classifierEntropy, double baseClassifierEntropy) {
		}
		@Override
		public String unit() {
			return "";
		}
		@Override
		public void clear() {
		}
		@Override
		public double getValue(Evaluation evaluation) {
		    double[] classFrequencies = evaluation.getClassPriors();
			double sumFrequencies = 0;
			for(int i=0;i<classFrequencies.length;i++)
				sumFrequencies += classFrequencies[i];
			for(int i=0;i<classFrequencies.length;i++)
				classFrequencies[i] /= sumFrequencies;
		    //calculate imbalance
		    double imbalanceNom = 0;
			double imbalanceDenom = 0;
			for(int i=0;i<classFrequencies.length;i++) {
				if(classFrequencies[i]==0)
					continue;
			    for(int j=0;j<classFrequencies.length;j++)
				    if(i!=j){
				    	if(classFrequencies[j]==0)
				    		continue;
				    	imbalanceNom += classFrequencies[i]*classFrequencies[j]*Math.abs(evaluation.truePositiveRate(i)-evaluation.truePositiveRate(j));
				    	imbalanceDenom += classFrequencies[i]*classFrequencies[j];
				    }
			}
			return imbalanceNom/imbalanceDenom;
		}
	}
}
