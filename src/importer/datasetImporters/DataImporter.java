package importer.datasetImporters;

import java.io.BufferedReader;
import java.io.File;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashMap;

import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;

public class DataImporter {
	public static Instances importDatabase(String path) throws Exception {
		/*if((new File(path+".arff").exists())) {
			Instances instances = importer.datasetImporters.ArffImporter.arffImporter(path+".arff");
			instances.setClassIndex(0);
			return instances;
		}*/
		HashMap <String, Integer> classes = new HashMap<String, Integer>();
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
				if(classIndex<0)
					classIndex = split.length+classIndex;
				if(classes.get(split[classIndex])==null)
					classes.put(split[classIndex], classes.size());
				size++;
			}
			br.close();
		}
		if(classes.size()>30) {
			classes.clear();
			classIndex = 0;
			BufferedReader br = Files.newBufferedReader(Paths.get(new File(path).getPath()));
			String line = null;
			String pending = "";
			while((line=br.readLine())!=null) {
				line = pending+line;
				String[] split = line.split("(\\s|\\,)+");
				if(length==0)
					length = split.length;
				if(classIndex<0)
					classIndex = split.length+classIndex;
				if(classes.get(split[classIndex])==null)
					classes.put(split[classIndex], classes.size());
			}
			br.close();
			
		}
		if(classes.size()>30)
			throw new RuntimeException("Could not identify class attribute");
		
		
		FastVector classValues = new FastVector(2);
		for(String dimensionName : classes.keySet())
			classValues.addElement(dimensionName);

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
				Instance instance = new Instance(length);
				for(int i=0;i<split.length;i++) {
					if(i==classIndex)
						continue;
					double w = 0;
					if(split[i].equalsIgnoreCase("m"))//male
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
					else if(split[i].isEmpty() || split[i].equals("?"))//unknown/empty attribute
						w = 0;
					else 
						w = Double.parseDouble(split[i]);
					if(i<classIndex) 
						instance.setValue(i+1, w);
					else if(i>classIndex)
						instance.setValue(i, w);
				}
				instance.setDataset(instances);
				instance.setClassValue(split[classIndex]);
				instances.add(instance);
			}
			br.close();
		}
		System.out.println("Generated "+size+" instances with "+classes.size()+" classes (class attribute index: "+classIndex+")");
		

		ArffSaver saver = new ArffSaver();
		saver.setInstances(instances);
		saver.setFile(new File (path+".arff"));
		saver.writeBatch();
		
		return instances;
	}

}