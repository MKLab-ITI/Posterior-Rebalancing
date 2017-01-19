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

public class USEMPImporter {

	public static Instances importDatabase(String path, String category) throws Exception {
		HashMap<String, Integer> words = new HashMap<String, Integer> ();
		{
			BufferedReader br = Files.newBufferedReader(Paths.get(new File(path+"messageTerms.csv").getPath()));
			String line = null;
			while((line=br.readLine())!=null) {
				String[] split = line.split("\\,\\s+");
				for(int i=1;i<split.length;i++) {
					String text = split[i].split("\\s+")[0];
					if(words.get(text)==null) 
						words.put(text, words.size());
				}
			}
			br.close();
		}
		//find classes
		HashMap<String, Integer> classes = new HashMap<String, Integer> ();
		int fieldId = -1;
		{
			BufferedReader br = Files.newBufferedReader(Paths.get(new File(path+"survey.csv").getPath()));
			String line = br.readLine();
			String[] fields = line.split("\\;");
			for(int i=1;i<fields.length;i++)
				if(fields[i].toLowerCase().contains(category.toLowerCase())) {
					fieldId = i;
					break;
				}
			while((line=br.readLine())!=null) {
				String[] split = line.split("\\;");
				String text = split[fieldId];
				if(classes.get(text)==null && !text.equalsIgnoreCase("?"))
					classes.put(text, classes.size());
			}
			br.close();
		}
		//generates user word samples
		HashMap<String, double[]> userInputs = new HashMap<String, double[]> ();
		{
			BufferedReader br = Files.newBufferedReader(Paths.get(new File(path+"messageTerms.csv").getPath()));
			String line = null;
			while((line=br.readLine())!=null) {
				String[] split = line.split("\\,\\s+");
				double[] userInput = new double[words.size()];
				for(int i=1;i<split.length;i++) {
					String[] text = split[i].split("\\s+");
					userInput[words.get(text[0])] = Double.parseDouble(text[1]);
				}
				userInputs.put(split[0].split("\\s+")[1], userInput);
			}
			br.close();
		}
		//initialize instances
		FastVector attributes  = new FastVector(words.size()+1);
		FastVector classValues = new FastVector(classes.size());
		for(String classValue : classes.keySet())
			classValues.addElement(classValue);
		attributes.addElement(new Attribute(path, classValues));
		for(String word : words.keySet())
			attributes.addElement(new Attribute(word));
		Instances instances = new Instances(path, attributes, userInputs.size());
		instances.setClassIndex(0);
		//apply class information on user samples
		{
			BufferedReader br = Files.newBufferedReader(Paths.get(new File(path+"survey.csv").getPath()));
			String line = br.readLine();//read first line and ignore it
			while((line=br.readLine())!=null) {
				String[] split = line.split("\\;");
				if(classes.get(split[fieldId])==null)
					continue;
				double[] userInput = userInputs.get(split[0]);
				Instance instance = new Instance(userInput.length+1);
				instance.setDataset(instances);
				for(int i=0;i<userInput.length;i++)
					instance.setValue(1+i, userInput[i]);
				instance.setClassValue(split[fieldId]);
				instances.add(instance);
			}
			br.close();
		}
		return instances;
	}
}
