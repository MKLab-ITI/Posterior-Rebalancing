package importer;
import java.util.HashMap;

import weka.core.Instance;
import weka.core.Instances;

/**
 * <h1>Compliance</h1>
 * This class implements functionality that generates compliant datasets using a stochastic approach.
 * For this process to work, thuis
 * @author Krasanakis Emmanouil
 */
public class Compliance {
	/**
	 * <h1>compliantInstances</h1>
	 * This function transforms the given <code>originalInstances</code> to the features space of
	 * <code>targetInstances</code> (e.g. transform <code>myPersonality = compliantInstances(myPersonality, USEMP)</code>
	 * to then use the new <code>myPersonality</code> dataset to train <code>USEMP</code>).<br/>
	 * The function {@link #classSimilarity} is required to correctly assess the mappings between class names.
	 * @param originalInstances
	 * @param targetInstances
	 * @return a {@link weka.core.Instances} object similar to originalInstances 
	 * 			but in the feature (i.e. attribute) space of targetInstances
	 * @throws Exception
	 */
	public static Instances compliantInstances(Instances originalInstances, Instances targetInstances) throws Exception {
		HashMap<Integer, Integer> valueMap = new HashMap<Integer, Integer>();
		for(int i=0;i<originalInstances.numClasses();i++) {
			String originalClassName = originalInstances.classAttribute().value(i);
			int maxSimilarPosition = 0;
			double maxSimilarity = 0;
			for(int j=0;j<targetInstances.numClasses();j++) {
				String targetClassName = targetInstances.classAttribute().value(j);
				double similarity = classSimilarity(originalClassName, targetClassName);
				if(similarity>=maxSimilarity) {
					maxSimilarity = similarity;
					maxSimilarPosition = j;
				}
			}
			if(maxSimilarity>=0.25) {
				//System.out.println(originalClassName + " -> "+targetInstances.classAttribute().value(maxSimilarPosition) +" (similarity: "+maxSimilarity+")");
				valueMap.put(i, maxSimilarPosition);
			}
			else
				System.err.println("Did not match class "+originalClassName);
		}
		//System.out.println("Generated value map");
		
		int n1 = originalInstances.numAttributes();
		int n2 = targetInstances.numAttributes();
		int originalIndex = originalInstances.classIndex();
		int targetIndex = targetInstances.classIndex();
		int targetClasses = targetInstances.classAttribute().numValues();
		
		double[][] classCounts1 = new double[targetClasses][n1];
		for(int i=0;i<originalInstances.numInstances();i++) {
			Instance originalInstance = originalInstances.instance(i);
			Integer cl = valueMap.get((int)originalInstance.classValue());
			if(cl!=null)
				for(int ii=0;ii<n1;ii++)
					if(ii!=originalIndex)
						classCounts1[cl][ii] += originalInstance.value(ii);
		}
		
		double[][] classCounts2 = new double[targetClasses][n2];
		for(int i=0;i<targetInstances.numInstances();i++) {
			Instance targetInstance = targetInstances.instance(i);
			Integer cl = (int) targetInstance.classValue();
			if(cl!=null)
				for(int jj=0;jj<n2;jj++)
					if(jj!=targetIndex)
						classCounts2[cl][jj] += targetInstance.value(jj);
		}
		
		double[][] tranformationMatrix = new double[n1][n2];
		for(int cl=0;cl<targetClasses;cl++) {
			for(int ii=0;ii<n1;ii++)
				if(ii!=originalIndex)
					for(int jj=0;jj<n2;jj++)
						if(jj!=targetIndex)
							tranformationMatrix[ii][jj] += classCounts1[cl][ii]*classCounts2[cl][jj];
		}
		classCounts2 = null;
		classCounts2 = null;
		/*for(int i=0;i<originalInstances.numInstances();i++) {
			Instance originalInstance = originalInstances.instance(i);
			for(int j=0;j<targetInstances.numInstances();j++) {
				Instance targetInstance = targetInstances.instance(j);
				if(valueMap.get((int)originalInstance.classValue())!=null && valueMap.get((int)originalInstance.classValue())==targetInstance.classValue()) 
					for(int ii=0;ii<n1;ii++)
						if(ii!=originalIndex)
						for(int jj=0;jj<n2;jj++)
							if(jj!=targetIndex)
								tranformationMatrix[ii][jj] += originalInstance.value(ii)*targetInstance.value(jj);
			}
		}*/
		//System.out.println("Created a transformation matrix with "+(n1*n2)+" cells");
		
		Instances instances = new Instances(targetInstances, 0);
		for(int i=0;i<originalInstances.numInstances();i++) {
			Instance originalInstance = originalInstances.instance(i);
			if(valueMap.get((int)originalInstance.classValue())==null)
				continue;
			Instance instance = new Instance(n2);
			double[] weights = new double[n2];
			double[] sum = new double[n2];
			for(int ii=0;ii<n1;ii++)
				if(ii!=originalIndex)
					for(int jj=0;jj<n2;jj++)
						if(jj!=targetIndex) {
							weights[jj] += tranformationMatrix[ii][jj]*originalInstance.value(ii);
							sum[jj] += tranformationMatrix[ii][jj];
						}
			for(int jj=0;jj<n2;jj++)
				if(sum[jj]!=0)
					instance.setValue(jj, weights[jj]/sum[jj]);
			instance.setDataset(instances);
			instance.setClassValue(valueMap.get((int)originalInstance.classValue()));
			instances.add(instance);
		}
		return instances;
	}
	/**
	 * <h1>classSimilarity</h1>
	 * @param s1 a class name
	 * @param s2 a class name
	 * @return The similarity between two class names.
	 * @deprecated This is an implementation directly fitted to perform compliance between specific datasets
	 * and should be changed to reflect actual similarity between class names.
	 */
	public static double classSimilarity(String s1, String s2) {
		String s1Flag = s1.toLowerCase();
		String s2Flag = s2.toLowerCase();
		String flag = s1Flag;
		if(flag.contains("gay") || flag.contains("lesb") || flag.contains("homo") || flag.contains("non-heter"))
			flag = "homosexual";
		else if(flag.contains("heter") || flag.contains("straight"))
			flag = "heterosexual";
		s1Flag = flag;
			
		flag = s2Flag;
		if(flag.contains("gay") || flag.contains("lesb") || flag.contains("homo") || flag.contains("non-heter"))
			flag = "homosexual";
		else if(flag.contains("heter") || flag.contains("straight"))
			flag = "heterosexual";
		s2Flag = flag;
		return s1Flag.equals(s2Flag)?1:0;
		//return 1.0-distance(s1.toLowerCase(), s2.toLowerCase(), s1.length(), s2.length())/(double)Math.max(s1.length(), s2.length());
	}
	/**
	 * <h1>distance</h1>
	 * Usage: <code>distance(s1.toLowerCase(), s2.toLowerCase(), s1.length(), s2.length())</code><br/>
	 * Upper bound is <code>Math.max(s1.length(), s2.length()</code>
	 * @param s1
	 * @param s2
	 * @param l1
	 * @param l2
	 * @return The Levenshtein distance between <code>s1,s2</code> as an <b>integer</b>.
	 */
	protected static int distance(String s1, String s2, int l1, int l2) {
		if(l1==0)
			return l2;
		if(l2==0)
			return l1;
		int d1 = distance(s1, s2, l1-1, l2)+1;
		int d2 = distance(s1, s2, l1, l2-1)+1;
		int d3 = distance(s1, s2, l1-1, l2-1);
		if(s1.charAt(l1-1)!=s2.charAt(l2-1))
			d3 += 1;
		int min = d3;
		if(d1<min)
			min = d1;
		if(d2<min)
			min = d2;
		return min;
	}
}
