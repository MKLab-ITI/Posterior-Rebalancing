package importer;

import java.io.BufferedReader;
import java.io.File;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;

import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;

class DataImporter {	
	public static class StrToDoubleConverter {
		private HashMap<String, Double> values = null;
		public double convert(String str) {
			if(values==null) {
				try{
					double ret = Double.parseDouble(str);
					return ret;
				}
				catch(Exception e) {
					values = new HashMap<String, Double>();
					values.put(str, 0.0);
					return 0;
				}
			}
			else {
				Double ret = values.get(str);
				if(ret==null) {
					ret = (double)values.size();
					values.put(str, ret);
				}
				return (double)ret;
			}
			
		}
	}
	
	public static Instances importDatabase(String path, boolean convertToBinaryDataset) throws Exception {
		/*if((new File(path+".arff").exists())) {
			Instances instances = importer.datasetImporters.ArffImporter.arffImporter(path+".arff");
			instances.setClassIndex(0);
			return instances;
		}*/
		HashMap <String, String> classes = new HashMap<String, String>();
		HashMap <String, Integer> classFrequencies = new HashMap<String, Integer>();
		int length = 0;
		int size = 0;
		int classIndex = -1;
		{
			BufferedReader br = Files.newBufferedReader(Paths.get(new File(path).getPath()));
			String line = null;
			String pending = "";
			while((line=br.readLine())!=null) {
				line = pending+line;
				String[] split = line.split("(\\s|\\,)+");
				if(length==0)
					length = split.length;
				if(split.length<length)
					break;
				if(classIndex<0)
					classIndex = split.length+classIndex;
				if(classes.get(split[classIndex])==null) {
					classes.put(split[classIndex], split[classIndex]);
					classFrequencies.put(split[classIndex], 1);
				}
				else
					classFrequencies.put(split[classIndex], classFrequencies.get(split[classIndex])+1);
				size++;
			}
			br.close();
		}
		if(classes.size()>30) {
			classes.clear();
			classFrequencies.clear();
			classIndex = 0;
			BufferedReader br = Files.newBufferedReader(Paths.get(new File(path).getPath()));
			String line = null;
			String pending = "";
			while((line=br.readLine())!=null) {
				line = pending+line;
				String[] split = line.split("(\\s|\\,)+");
				if(length==0)
					length = split.length;
				if(split.length<length)
					break;
				if(classIndex<0)
					classIndex = split.length+classIndex;
				if(classes.get(split[classIndex])==null) {
					classes.put(split[classIndex], split[classIndex]);
					classFrequencies.put(split[classIndex], 1);
				}
				else
					classFrequencies.put(split[classIndex], classFrequencies.get(split[classIndex])+1);
			}
			br.close();
			
		}
		if(classes.size()>30)
			throw new RuntimeException("Could not identify class attribute");
		
		FastVector classValues;
		if(convertToBinaryDataset) {
			String minimumClass = "";
			int minimumFrequency = Integer.MAX_VALUE;
			for(String className : classes.keySet())
				if(classFrequencies.get(className)<=minimumFrequency) {
					minimumClass = className;
					minimumFrequency = classFrequencies.get(className);
				}
			for(String className : new ArrayList<String>(classes.keySet())) {
				if(className.equals(minimumClass))
					classes.put(className, minimumClass);
				else
					classes.put(className, "other");
			}
			classValues = new FastVector(2);
			classValues.addElement("other");
			classValues.addElement(minimumClass);
		}
		else {
			classValues = new FastVector(classes.size());
			for(String dimensionName : classes.keySet())
				classValues.addElement(dimensionName);
		}

		ArrayList<StrToDoubleConverter> converters = new ArrayList<StrToDoubleConverter>();
		for(int i=0;i<length+1;i++)
			converters.add(new StrToDoubleConverter());
		FastVector attributes  = new FastVector(length+1);
		attributes.addElement(new Attribute(path, classValues));
		for(int i=0;i<length;i++)
			attributes.addElement(new Attribute(""+i));
		Instances instances = new Instances(path, attributes, size);
		instances.setClassIndex(0);
		{
			BufferedReader br = Files.newBufferedReader(Paths.get(new File(path).getPath()));
			String line; //br.readLine();//read first line and ignore it
			String pending = "";
			while((line=br.readLine())!=null) {
				line = pending+line;
				String[] split = line.split("(\\s|\\,)+");
				if(split.length<length)
					break;
				Instance instance = new Instance(length);
				for(int i=0;i<split.length;i++) {
					if(i==classIndex)
						continue;
					double w = 0;
					/*if(split[i].equalsIgnoreCase("m"))//male
						w = 1;
					else if(split[i].equalsIgnoreCase("t"))//true
						w = 1;
					else if(split[i].equalsIgnoreCase("f"))//female/false
						w = 0;
					else if(split[i].equalsIgnoreCase("y"))//yes
						w = 1;
					else if(split[i].equalsIgnoreCase("n"))//no
						w = 0;
					else if(split[i].equalsIgnoreCase("i"))//for abalone dataset only
						w = 2;
					else*/ if(split[i].isEmpty() || split[i].equals("?"))//unknown/empty attribute
						w = 0;
					else 
						w = converters.get(i).convert(split[i]);
					if(i<classIndex) 
						instance.setValue(i+1, w);
					else if(i>classIndex)
						instance.setValue(i, w);
				}
				instance.setDataset(instances);
				instance.setClassValue(classes.get(split[classIndex]));
				instances.add(instance);
			}
			br.close();
		}
		//System.out.println("Generated "+size+" instances with "+classValues.size()+" classes (class attribute index: "+classIndex+")");
		

		ArffSaver saver = new ArffSaver();
		saver.setInstances(instances);
		saver.setFile(new File (path+".arff"));
		saver.writeBatch();
		
		return instances;
	}

}