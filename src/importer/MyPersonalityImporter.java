package importer;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.HashMap;

import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

class MyPersonalityImporter {
	protected static int countLines(File file) throws Exception {
		int count = 0;
		BufferedReader br = new BufferedReader(new FileReader(file.getPath()));
		while(br.readLine()!=null)
			count++;
		br.close();
		return count;
	}
	
	public static Instances importLDA(String path) throws Exception {
		//find dimensions
		int totalSamples = 0;
		HashMap<String, Integer> dimensionIds = new HashMap<String, Integer>();
		File folder = new File(path);
		for(File file : folder.listFiles()) {
			if(file.isDirectory()) 
				continue;
			if(file.getName().contains("ids"))
				continue;
			String dimensionName = file.getName();
			if(dimensionIds.get(dimensionName)==null)
				dimensionIds.put(dimensionName, dimensionIds.size());
			totalSamples += countLines(file)-1;
		}
		//generate weka class values for dimensions
		FastVector classValues = new FastVector(dimensionIds.size());
		for(String dimensionName : dimensionIds.keySet())
			classValues.addElement(dimensionName);

		Instances instances = null;
		//import each dimension
		for(File file : folder.listFiles()) {
			if(file.isDirectory()) 
				continue;
			if(file.getName().contains("ids"))
				continue;
			//read first line a priori to find and construct dimensions
			BufferedReader br = new BufferedReader(new FileReader(file.getPath()));
			String line = br.readLine();
			if(line==null)
				continue;
			//initialize instances according to number of descriptors
			int LDAdimensions = line.split("[\\s\\\"]*\\,[\\s\\\"]*").length;
			if(instances==null) {
				FastVector attributes  = new FastVector(LDAdimensions);
				attributes.addElement(new Attribute(path, classValues));
				for(int i=1;i<LDAdimensions;i++)
					attributes.addElement(new Attribute("LDA"+i));
				instances = new Instances(path, attributes, totalSamples);
				instances.setClassIndex(0);
			}
			while(line != null) {
				String[] elements = line.split("[\\s\\\"]*\\,[\\s\\\"]*");
				if(elements.length<=1)
					continue;
				Instance instance = new Instance(elements.length);
				instance.setDataset(instances);
				for(int i=1;i<elements.length;i++) 
					instance.setValue(i, Double.parseDouble(elements[i]));
				instance.setValue(0, file.getName());
				instances.add(instance);
				line = br.readLine();
			}
			br.close();
		}

		return instances;
	}
	
	
}
