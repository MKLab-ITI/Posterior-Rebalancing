package algorithms.baseClassifiers;
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
 * <h1>Clusterer</h1>
 * This weka {@link Classifier} splits training points into clusters according to a repeating binary split.
 * Clusters are further split so that each cluster has members belonging to only one class.<br/>
 * Options can either be parsed through the constructor or through {@link #setOptions(String[])};<br/>
 * <h2>-wei[ghtRatePow]</h2>
 * Exponentiates weight rates.
 * <h2>-rad[iusStrictness]</h2>
 * Values closer to <code>0</code> make cluster radius estimation more lax. Values should be normally near
 * to <code>1-Var(cluster)</code>. Extremely small values tend to just be similar to weighted KNN between
 * cluster centers. Default value is <code>0.05</code>. 
 * <h2>-classNorm[alizerPow]</h2>
 * Weight normalization for each class may have its denominator exponated with this value. Its default value
 * is <code>1</code>, so as to ignore this feature.
 * <h2>-debug</h1>
 * Outputs split process information to <code>System.out</code>. (After enabling, it cannot be disabled.)
 * 
 * @author Krasanakis Emmanouil
 */
public class Clusterer extends Classifier implements Serializable {
	private static final long serialVersionUID = 2901702435471874617L;
	
	protected static double[] mean(Collection<double[]> points) {
		double[] ret = null;
		int count = 0;
		for(double[] point : points) {
			if(ret==null)
				ret = new double[point.length];
			add(ret, point);
			count++;
		}
		multiply(ret, 1.0/count);
		return ret;
	}
	protected static void add(double[] to, double[] from) {
		for(int i=0;i<to.length;i++)
			to[i] += from[i];
	}
	protected static void weightedAdd(double[] to, double[] from, double multiply) {
		for(int i=0;i<to.length;i++)
			to[i] += from[i]*multiply;
	}
	protected static void multiply(double[] to, double val) {
		for(int i=0;i<to.length;i++)
			to[i] *= val;
	}

	protected static double similarity(double[] v1, double[] v2) throws Exception {
		double ret = 0;
		if(v1.length!=v2.length)
			throw new Exception("Different vector lengths");
		for(int i=0;i<v1.length;i++)
			ret += v1[i]*v2[i];
		return ret / v1.length;
	}
	protected double[] toInputArray(Instance instance) {
		double[] ret = new double[instance.numAttributes()-1]; 
		for(int m=0;m<classPosition;m++) 
			ret[m] = instance.value(m);
		for(int m=classPosition;m<ret.length;m++) 
			ret[m] = instance.value(1+m);
		return ret;
	}

	private double radiusStrictness = 0.05;
	private double classVectorNormalizerPow = 1;//is 1 normally
	private double weightRate = 0;
	public boolean debug = false;
	protected transient HashMap<double[], double[]> points;//donnot serialize this
	protected HashMap<double[], double[]> centroids;
	protected HashMap<double[], Double> centroidWeights;
	protected HashMap<double[], Double> centroidLimit;
	private int classPosition;
	protected int outputSize;
	
	public Clusterer() throws Exception {
		this(weka.core.Utils.splitOptions("-weightRatePow 0 -radiusStrictness 0.05 -classNormalizerPow 1"));
	}
	public Clusterer(String[] options) throws Exception {
		setOptions(options);
	}

	@Override
	public void setOptions(String[] options) throws Exception {
		super.setOptions(options);
		for(int i=0;i<options.length;i++) 
			options[i] = options[i].toLowerCase();
		for(int i=0;i<options.length;i++) 
			if(options[i].startsWith("-wei")) {
				i++;
				double pow = Double.parseDouble(options[i]);
				if(pow==0)
					weightRate = 1;
				else
					weightRate = 1+Math.signum(pow)*Math.pow(0.1, Math.abs(pow));
			}
			else if(options[i].startsWith("-rad")) {
				i++;
				radiusStrictness = Double.parseDouble(options[i]);
			}
			else if(options[i].startsWith("-classnorm")) {
				i++;
				classVectorNormalizerPow = Double.parseDouble(options[i]);
			}
			else if(options[i].startsWith("-debug")) {
				debug = true;
			}
			else
				throw new Exception("Unknown or malformed clusterer options");
	}
	
	@Override
	public void buildClassifier(Instances instances) throws Exception {
		points = new HashMap<double[], double[]>();
		centroids = new HashMap<double[], double[]>();
		centroidWeights = new HashMap<double[], Double>();
		centroidLimit = new HashMap<double[], Double>();
		
		outputSize = instances.classAttribute().numValues();
		classPosition = instances.classIndex();
		
		@SuppressWarnings("unchecked")
		Enumeration<Instance> enumeration = instances.enumerateInstances();
		while(enumeration.hasMoreElements()) {
			Instance instance = enumeration.nextElement();
			double[] outArray = new double[outputSize];
			outArray[(int)instance.classValue()] = 1;
			points.put(toInputArray(instance), outArray);
		}
		
		
		//post-processing
		initialGroup();
		HashMap<double[], Integer> groupIncohesiveness = new HashMap<double[], Integer>();
		boolean changed = true;
		while(changed) {
			changed = false;
			double[] pendingSplit = null;
			int maxIncohesiveness = 5;
			for(Entry<double[], Collection<double[]>> group : groups.entrySet()) {
				Integer incohesiveness = groupIncohesiveness.get(group.getKey());
				if(incohesiveness==null)
					groupIncohesiveness.put(group.getKey(), incohesiveness = getIncohesiveCount(group.getValue()));
				if(incohesiveness>maxIncohesiveness) {
					pendingSplit = group.getKey();
					maxIncohesiveness = incohesiveness;
					break;
				}
			}
			changed = pendingSplit!=null;
			if(pendingSplit!=null) {
				if(debug)
					System.out.print("Splitting group with "+groups.get(pendingSplit).size()+" elements ("+maxIncohesiveness+" are incohesive)... ");
				groupIncohesiveness.remove(pendingSplit);
				if(!splitGroup(pendingSplit)) {
					groupIncohesiveness.put(pendingSplit, 0);
					if(debug)
						System.out.println("clustering not possible; will split according to class labels.");
				}
			}
		}
		changed = true;
		while(changed) {
			changed = false;
			for(Entry<double[], Collection<double[]>> group : groups.entrySet()) 
				if(getIncohesiveCount(group.getValue())>0) {
					splitClassGroup(group.getKey());
					changed = true;
					break;
				}
		}
		for(Entry<double[], Collection<double[]>> group : groups.entrySet()) {
			double[] classVector = new double[outputSize];
			double totalSimilarity = 0;
			double[] groupCenterPoint = group.getKey();
			double minSimilarity = Double.POSITIVE_INFINITY;
			for(double[] point : group.getValue()) {
				double similarity = similarity(groupCenterPoint, point);
				weightedAdd(classVector, points.get(point), similarity);
				totalSimilarity += similarity;
				if(similarity<minSimilarity)
					minSimilarity = similarity;
			}
			if(totalSimilarity!=0)
				multiply(classVector, Math.pow(totalSimilarity, -classVectorNormalizerPow));
			centroidLimit.put(groupCenterPoint, totalSimilarity/group.getValue().size()*radiusStrictness);
			centroids.put(groupCenterPoint, classVector);

			if(weightRate!=1) {
				double weight = 1;
				for(Entry<double[], double[]> point : points.entrySet()) {
					double similarity = similarity(groupCenterPoint, point.getKey());
					if(similarity>centroidLimit.get(groupCenterPoint)) {
						double same = 0;
						for(int i=0;i<classVector.length;i++)
							same += classVector[i]*point.getValue()[i];
						if(same>0)
							weight /= weightRate;
						else 
							weight *= weightRate;
					}
				}
				centroidWeights.put(group.getKey(), weight);
			}
		}
		if(debug)
			System.out.println("Process generated "+groups.size()+" groups");
	}
	
	@Override
	public double[] distributionForInstance(Instance instance) throws Exception {
		double[] ret = new double[outputSize];
		double[] input = toInputArray(instance);
		double similaritySum = 0;
		for(Entry<double[], double[]> centroid : centroids.entrySet()) {
			if(!isWithinCentroid(centroid.getKey(), input))
				continue;
			Double similarityMultiplier = centroidWeights.get(centroid.getKey());
			if(similarityMultiplier==null)
				similarityMultiplier = 1.0;
			double similarity = similarity(input, centroid.getKey())*similarityMultiplier;
			double[] tmp = centroid.getValue().clone();
			multiply(tmp, similarity);
			similaritySum += similarity;
			add(ret, tmp);
		}
		if(similaritySum!=0)
			multiply(ret, 1.0/similaritySum);
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
	
	private boolean isWithinCentroid(double[] centroid, double[] point) throws Exception {
		if(centroidLimit.get(centroid)==null)
			return true;
		return similarity(centroid, point)>centroidLimit.get(centroid);
	}
	private HashMap<double[], Collection<double[]>> groups = new HashMap<double[], Collection<double[]>> ();
	private void initialGroup() {
		double[] center = mean(points.keySet());
		groups.put(center, points.keySet());
	}
	private int getIncohesiveCount(Collection<double[]> groupPoints) {
		double[] out = null;
		int count = 0;
		for(double[] point : groupPoints) {
			if(out==null)
				out = points.get(point);
			double[] pointValue = points.get(point);
			boolean similar = true;
			for(int i=0;i<out.length;i++)
				if(out[i]!=pointValue[i]) {
					similar = false;
					break;
				}
			if(!similar)
				count++;
		}
		if(count>groupPoints.size()/2)
			count = groupPoints.size()-count;
		return count;
	}
	private boolean splitClassGroup(double[] group) {
		ArrayList<double[]> classGroup = new ArrayList<double[]>();
		ArrayList<double[]> complementaryGroup = new ArrayList<double[]>();
		Collection<double[]> groupPoints = groups.get(group);
		double[] out = null;
		for(double[] point : groupPoints) {
			if(out==null)
				out = points.get(point);
			double[] pointValue = points.get(point);
			boolean similar = true;
			for(int i=0;i<out.length;i++)
				if(out[i]!=pointValue[i]) {
					similar = false;
					break;
				}
			if(similar)
				classGroup.add(point);
			else
				complementaryGroup.add(point);
		}
		if(complementaryGroup.isEmpty())
			return false;
		groups.remove(group);
		groups.put(mean(classGroup), classGroup);
		groups.put(mean(complementaryGroup), complementaryGroup);
		return true;
	}
	private boolean splitGroup(double[] group) throws Exception {
		Collection<double[]> groupPoints = groups.get(group);
		
		ArrayList<double[]> points1 = new ArrayList<double[]> ();
		ArrayList<double[]> points2 = new ArrayList<double[]> ();
		for(double[] point : groupPoints) {
			if(points1.size()<points2.size())
				points1.add(point);
			else
				points2.add(point);
		}
		int prevChanged = Integer.MAX_VALUE;
		int changed = groupPoints.size()+2;
		double[] center1 = mean(points1);
		double[] center2 = mean(points2);
		while(changed<prevChanged) {
			prevChanged = changed;
			changed = 0;
			for(double[] point : groupPoints) {
				double similarity1 = similarity(point, center1);
				double similarity2 = similarity(point, center2);
				if(similarity1>similarity2 && !points1.contains(point)) {
					points2.remove(point);
					points1.add(point);
					changed++;
				}
				else if(similarity1<similarity2 && !points2.contains(point)) {
					points1.remove(point);
					points2.add(point);
					changed++;
				}
			}
			if(points1.isEmpty() || points2.isEmpty())
				return false;
			center1 = mean(points1);
			center2 = mean(points2);
		}
		if(debug)
			System.out.println(" into two groups with "+points1.size()+" and "+points2.size()+" points respectively");
		
		groups.remove(group);
		groups.put(center1, points1);
		groups.put(center2, points2);
		return true;
	}
	
}