package algorithms.baseClassifiers;

import java.io.Serializable;
import java.util.ArrayList;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

/**
 * <h1>ImprovedSmoothKNN</h1>
 * This Weka {@link Classifier} performs fast kNN computations without using slower Weka structures.
 * Voting is performed while taking object {@link #dot} similarity into account.<br/>
 * There are not options to be set.
 * 
 * @author Emmanouil Krasanakis
 */
public class ImprovedSmoothKNN extends Classifier implements Serializable { 
	private static final long serialVersionUID = 5994345396055948907L;
	private int neighbors = 5;
	private Instances instances;
	private int classIndex;
	private double[] norms;
	private int[] numberAttrs;
	private int[] nominalAttrs;
			
	public ImprovedSmoothKNN() {
	}
	@Override
	public void buildClassifier(Instances instances) throws Exception {
		this.instances = instances;
		classIndex = instances.classIndex();
		ArrayList<Integer> numberAttrsList = new ArrayList<Integer>();
		ArrayList<Integer> nominalAttrsList = new ArrayList<Integer>();
		for(int i=0;i<classIndex;i++) 
			if(instances.attribute(i).isNumeric()) 
				numberAttrsList.add(i);
			else
				nominalAttrsList.add(i);
		for(int i=classIndex+1;i<instances.numAttributes();i++) 
			if(instances.attribute(i).isNumeric()) 
				numberAttrsList.add(i);
			else
				nominalAttrsList.add(i);
		numberAttrs = new int[numberAttrsList.size()];
		for(int i=0;i<numberAttrs.length;i++)
			numberAttrs[i] = numberAttrsList.get(i);
		nominalAttrs = new int[nominalAttrsList.size()];
		for(int i=0;i<nominalAttrs.length;i++)
			nominalAttrs[i] = nominalAttrsList.get(i);

		norms = new double[instances.numInstances()];
		for(int n=0;n<instances.numInstances();n++)
			norms[n] = Math.sqrt(dot(instances.instance(n), instances.instance(n)));
			
	}
	
	/**
	 * <h1>dot</h1>
	 * This function supports both numeric and nominal atttributes. Nominal attributes contribute only as being
	 * identical or not similar.
	 * @param instance
	 * @param test
	 * @return the dot product between two {@link weka.core.Instance} objects, excluding NaN fields
	 * (therefore will also exclude unknown class fields)
	 */
	protected double dot(Instance instance, Instance test) {
		double sim = 0;
		for(int i : numberAttrs)
			if(!Double.isNaN(instance.value(i)*test.value(i)))
				sim += instance.value(i)*test.value(i);
		for(int i : nominalAttrs)
			sim += instance.value(i)==test.value(i)?1:0;
		return sim;
	}
	
	@Override
	public double[] distributionForInstance(Instance instance) throws Exception {
		double[] output = new double[instance.classAttribute().numValues()];
		double[] similarities = new double[instances.numInstances()];
		ArrayList<Integer> previouslySelected = new ArrayList<Integer>();
		double norm = Math.sqrt(dot(instance,instance));
		for(int k=0;k<neighbors;k++) {
			if(k==0) 
				for(int n=0;n<instances.numInstances();n++) 
					similarities[n] = dot(instance, instances.instance(n))/norm/norms[n]*instances.instance(n).weight();
			int selected = -1; 
			double similarity = 0;
			for(int n=0;n<instances.numInstances();n++) 
				if(similarities[n]>similarity && !previouslySelected.contains(n)){
					selected = n;
					similarity = similarities[n];
				}
			if(selected==-1) {
				//System.err.println("Could not find neighbors");
				break;
			}
			previouslySelected.add(selected);
			int classValue = (int)instances.instance(selected).classValue();
			output[classValue] += similarity;
		}
		double normalize = 0;
		for(int i=0;i<output.length;i++)
			normalize += output[i];
		if(normalize!=0)
			for(int i=0;i<output.length;i++)
				output[i] /= normalize;
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
