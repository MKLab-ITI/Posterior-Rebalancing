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
import weka.core.SparseInstance;
import weka.core.converters.ArffSaver;

public class UCIImporter {
	public static Instances importDatabase(String path, String[] classes, int classIndex) throws Exception {
		if((new File(path+".arff").exists())) {
			Instances instances = importer.datasetImporters.ArffImporter.arffImporter(path+".arff");
			instances.setClassIndex(0);
			return instances;
		}
		
		
		FastVector classValues = new FastVector(2);
		for(String dimensionName : classes)
			classValues.addElement(dimensionName);
		int size = 0;
		HashMap<String, Integer> words = new HashMap<String, Integer> ();
		{
			BufferedReader br = Files.newBufferedReader(Paths.get(new File(path).getPath()));
			String line = null;
			String pending = "";
			while((line=br.readLine())!=null) {
				line = pending+line;
				String[] split = line.replaceAll("[^\\w\\s]"," ").toLowerCase().split("(\\s|\\\t)+");
				int cl = classIndex;
				if(cl<0)
					cl += split.length;
				boolean foundClass = false;
				for(String str : classes)
					if(str.equals(split[cl])){
						foundClass = true;
						break;
					}
				if(!foundClass) {
					pending = line;
					continue;
				}
				pending = "";
				for(int i=0;i<split.length;i++) 
				if(i!=cl){
					String text = split[i];
					if(words.get(text)==null && text.length()>=2) 
						words.put(text, words.size());
				}
				size++;
			}
			br.close();
		}

		FastVector attributes  = new FastVector(words.size()+1);
		attributes.addElement(new Attribute(path, classValues));
		for(String word : words.keySet())
			attributes.addElement(new Attribute(word));
		Instances instances = new Instances(path, attributes, size);
		instances.setClassIndex(0);
		size = 0;
		System.out.println("Extracted words for "+size+" entries");
		{
			int instanceSize = words.size()+1;
			BufferedReader br = Files.newBufferedReader(Paths.get(new File(path).getPath()));
			String line = br.readLine();//read first line and ignore it
			String pending = "";
			while((line=br.readLine())!=null) {
				line = pending+line;
				String[] split = line.replaceAll("[^\\w\\s]"," ").toLowerCase().split("(\\s|\\t)+");
				int cl = classIndex;
				if(cl<0)
					cl += split.length;
				boolean foundClass = false;
				for(String str : classes)
					if(str.equals(split[cl])){
						foundClass = true;
						break;
					}
				if(!foundClass) {
					pending = line;
					continue;
				}
				pending = "";
				Instance instance = new SparseInstance(instanceSize);
				instance.setDataset(instances);
				for(int i=1;i<instanceSize;i++)
					instance.setValue(i, 0);
				for(int i=0;i<split.length;i++)
					if(i!=cl && split[i].length()>=2) {
						Integer w = words.get(split[i]);
						if(w!=null)
							instance.setValue(w+1, 1);
					}
				instance.setClassValue(split[cl]);
				instances.add(instance);
				size++;
			}
			br.close();
		}
		System.out.println("Generated "+size+" instances");
		

		ArffSaver saver = new ArffSaver();
		saver.setInstances(instances);
		saver.setFile(new File (path+".arff"));
		saver.writeBatch();
		
		return instances;
	}

}
