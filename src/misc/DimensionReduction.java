package misc;
import java.util.ArrayList;
import java.util.Enumeration;

import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

public class DimensionReduction {
	public static Instances reduce(Instances original, int dimensions) throws Exception {
		if(original==null)
			return null;
		if(original.numAttributes()<dimensions-1)
			return original;
		int classPosition = original.classIndex();
		//CALCULATE VARIANCE
		double[] attributeVariance = new double[original.numAttributes()];
		{
			double[] attributeSum = new double[original.numAttributes()];
			@SuppressWarnings("unchecked")
			Enumeration<Instance> enumeration = original.enumerateInstances();
			while(enumeration.hasMoreElements()) {
				Instance instance = enumeration.nextElement();
				for(int i=0;i<attributeSum.length;i++) 
					if(i!=classPosition){
						double value = instance.value(i);
						attributeSum[i] += value;
						attributeVariance[i] += value*value;
					}
			}
			int n = original.numInstances();
			for(int i=0;i<attributeVariance.length;i++)
				attributeVariance[i] = attributeVariance[i]/n-Math.pow(attributeSum[i]/n, 2);
		}
		
		//FIND MAX ATTRIBUTES
		ArrayList<Integer> maxAttributes = new ArrayList<Integer>();
		while(dimensions>0) {
			int maxPos = -1;
			double maxVariance = 0;
			for(int pos=0;pos<attributeVariance.length;pos++) 
				if(attributeVariance[pos]>maxVariance && !maxAttributes.contains(pos)){
					maxPos = pos;
					maxVariance = attributeVariance[pos];
				}
			if(maxPos==-1)
				throw new Exception("Not enough original attributes");
			dimensions--;
			maxAttributes.add(maxPos);
		}

		//CREATE ATTRIBUTS AND INSTANCES
		FastVector attributes  = new FastVector(maxAttributes.size()+1);
		attributes.addElement(original.classAttribute());
		for(int i=0;i<maxAttributes.size();i++)
			attributes.addElement(original.attribute(maxAttributes.get(i)));
		Instances instances = new Instances(original.relationName(), attributes, original.numInstances());
		instances.setClassIndex(0);
		
		@SuppressWarnings("unchecked")
		Enumeration<Instance> enumeration = original.enumerateInstances();
		while(enumeration.hasMoreElements()) {
			Instance originalInstance = enumeration.nextElement();
			Instance instance = new Instance(maxAttributes.size()+1);
			for(int i=0;i<maxAttributes.size();i++)
				instance.setValue(i+1, originalInstance.value(maxAttributes.get(i)));
			instance.setDataset(instances);
			instance.setClassValue(originalInstance.classValue());
			instances.add(instance);
		}
		
		return instances;
	}
}
