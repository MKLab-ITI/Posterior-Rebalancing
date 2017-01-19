package algorithms.implemtations;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.Map.Entry;

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

/**
 * <h1>SmoothKNN</h1>
 * This weka {@link Classifier} performs classification using a mixture of Weighted-KNN and Informativeness-KNN.<br/>
 * Weight calculation options can either be parsed through the constructor or through {@link #setOptions(String[])};<br/>
 * <h1>-neighbors</h1>
 * The number of neigbors used to classify each instance (default value is 5, while 3 is usually used for small
 * and 11 for extremely large datasets).
 * <h2>-powSim[ilarity]</h2>
 * Exponent of the similarity in node weighting (0 to ignore similarity in weighting, default value is 1).
 * <h2>-powWei[ght]</h2>
 * Exponent of the importance in node weighting (0 to ignore node importance in weighting, default value is 1).
 * <h2>-sim[ilarity]</h2>
 * Selects the way input similarity and node weights are computed (default value is informativeness); <br/>
 * <code>dot</code> : uses the dot product (is equivalent to using cosine similarity, but less computationally-intensive) <br/>
 * <code>euc[lidean]</code> : uses the Euclidean probability exp(-|x1-x2|) to calculate similarity <br/>
 * <code>info[rmativeness]</code> : uses the Euclidean probability but also assesses points using their informativeness <br/>
 * <br/>
 * Node weights are calculated as the sum of their similarities for <code>dot</code> similarity,
 * as the multiplication of their Euclidean probability for <code>euclidean</code> similarity or
 * as the multiplication of their informativeness (i.e. <code>-log(1-e)*e</code> with Euclidean probability e)
 * for informativeness similarity.
 * 
 * @author Krasanakis Emmanouil
 */
public class SmoothKNN extends Classifier implements Serializable {
	private static final long serialVersionUID = 829246676038627545L;
	protected static enum SimilarityType {DOT, EUCLIDEAN, INFORMATIVENESS};
	private HashMap<double[], double[]> data;
	private HashMap<double[], Double> pointEffectiveness;
	private int outputSize;
	private int classPosition;
	private double proximityWeightBalance;
	private double similarNeighborWeightBalanance;
	private SimilarityType similarityType;
	private int neighbors;

	protected static double outputSimilarity(double[] v1, double[] v2) throws Exception {
		double ret = 0;
		if(v1.length!=v2.length)
			throw new Exception("Different vector lengths");
		for(int i=0;i<v1.length;i++)
			if(!Double.isNaN(v1[i]*v2[i]))
				ret += v1[i]*v2[i];
		return ret; // v1.length;
	}
	protected  double similarity(double[] v1, double[] v2) throws Exception {
		if(similarityType==SimilarityType.DOT)
			return outputSimilarity(v1, v2) / Math.sqrt(outputSimilarity(v1, v1)*outputSimilarity(v2, v2));
		if(v1.length!=v2.length)
			throw new Exception("Different vector lengths");
		double ret = 0;
		for(int i=0;i<v1.length;i++) 
			if(!Double.isNaN(v1[i]-v2[i]))
				ret += (v1[i]-v2[i])*(v1[i]-v2[i]);
		return Math.exp(-Math.sqrt(ret / v1.length));
	}
	protected  double[] toInputArray(Instance instance) {
		double[] ret = new double[instance.numAttributes()-1]; 
		for(int m=0;m<classPosition;m++)
			ret[m] = instance.value(m); 
		for(int m=classPosition;m<ret.length;m++)
			ret[m] = instance.value(1+m);
		return ret;
	}
	public SmoothKNN() throws Exception {
		this(weka.core.Utils.splitOptions("-neighbors 5 -similarity informativeness -powSim 1 -powWeight 0"));
	}
	public SmoothKNN(String[] options) throws Exception{
		setOptions(options);
	}
	@Override
	public void setOptions(String[] options) throws Exception {
		super.setOptions(options);
		for(int i=0;i<options.length;i++) 
			options[i] = options[i].toLowerCase();
		for(int i=0;i<options.length;i++) 
			if(options[i].startsWith("-powsim")) {
				i++;
				proximityWeightBalance = Double.parseDouble(options[i]);
			}
			else if(options[i].startsWith("-powwei")) {
				i++;
				similarNeighborWeightBalanance = Double.parseDouble(options[i]);
			}
			else if(options[i].startsWith("-nei")) {
				i++;
				neighbors = Integer.parseInt(options[i]);
			}
			else if(options[i].startsWith("-sim")) {
				i++;
				if(options[i].startsWith("dot"))
					similarityType = SimilarityType.DOT;
				else if(options[i].startsWith("info"))
					similarityType = SimilarityType.INFORMATIVENESS;
				else if(options[i].startsWith("euc"))
					similarityType = SimilarityType.EUCLIDEAN;
				else
					throw new Exception("Invalid -similarity option");
			}
			else
				throw new Exception("Unknown or malformed smoothing options");
	}
	
	protected ArrayList<double[]> getNearestPoints(double[] center, Collection<double[]> points, int k) throws Exception {
		ArrayList<double[]> discoveredPoints = new ArrayList<double[]>();
		discoveredPoints.add(center);
		HashMap<double[], Double> similarities = new HashMap<double[], Double>();
		for(int repeat=0;repeat<k;repeat++) {
			double maxSimilarity = Double.NEGATIVE_INFINITY;
			double[] discovered = null;
			for(double[] point : points) {
				Double similarity = similarities.get(point);
				if(similarity==null)
					similarities.put(point, similarity = similarity(center, point));
				if(!discoveredPoints.contains(point) && similarity>maxSimilarity) {
					maxSimilarity = similarity;
					discovered = point;
				}
			}
			if(discovered!=null)
				discoveredPoints.add(discovered);
		}
		discoveredPoints.remove(center);
		return discoveredPoints;
	}
	
	@Override
	public void buildClassifier(Instances instances) throws Exception {
		data = new HashMap<double[], double[]> ();
		pointEffectiveness = new HashMap<double[], Double> ();
		outputSize = instances.classAttribute().numValues();
		classPosition = instances.classIndex();

		@SuppressWarnings("unchecked")
		Enumeration<Instance> enumeration = instances.enumerateInstances();
		while(enumeration.hasMoreElements()) {
			Instance instance = enumeration.nextElement();
			double[] outArray = new double[outputSize];
			outArray[(int)instance.classValue()] = 1;
			data.put(toInputArray(instance), outArray);
		}

		if(similarNeighborWeightBalanance!=0)
		for(Entry<double[], double[]> point1 : data.entrySet()) {
			for(double[] point2 : getNearestPoints(point1.getKey(), data.keySet(), neighbors)) {
				double outputSimilarityValue = outputSimilarity(point1.getValue(), data.get(point2));
				if(outputSimilarityValue==0)
					continue;
				double inputSimilarityValue = similarity(point1.getKey(), point2);
				if(similarityType==SimilarityType.INFORMATIVENESS)
					inputSimilarityValue = -Math.log(1-inputSimilarityValue*0.95)*inputSimilarityValue/5;
				//if(inputSimilarityValue<0)
					//throw new Exception("Illegal point value");
				Double effectiveness = pointEffectiveness.get(point2);
				if(similarityType==SimilarityType.DOT) {
					if(effectiveness==null)
						effectiveness = 0.0;
					effectiveness += inputSimilarityValue*outputSimilarityValue;
				}
				else {
					if(effectiveness==null)
						effectiveness = 1.0;
					effectiveness *= Math.pow(inputSimilarityValue, outputSimilarityValue);
				}

				pointEffectiveness.put(point2, effectiveness);
			}
		}
	}
	
	@Override
	public double[] distributionForInstance(Instance instance) throws Exception {
		double[] output = new double[outputSize];
		double[] input = toInputArray(instance);
		//int count = 0;
		for(double[] point : getNearestPoints(input, data.keySet(), neighbors)) {
			double similarity = Math.pow(similarity(point, input), proximityWeightBalance);
			Double effectiveness = pointEffectiveness.get(point);
			if(effectiveness==null || similarNeighborWeightBalanance==0)
				effectiveness = 1.0;
			else
				effectiveness = Math.pow(effectiveness, similarNeighborWeightBalanance);
			double[] pointValue = data.get(point);
			for(int i=0;i<output.length;i++) 
				output[i] += similarity*effectiveness*pointValue[i];
			//count++;
		}
		double norm = 0;
		for(int i=0;i<output.length;i++)
			norm += output[i];
		if(norm!=0)
			for(int i=0;i<output.length;i++)
				output[i] /= norm;
		return output;
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