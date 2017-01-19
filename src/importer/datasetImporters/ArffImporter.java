package importer.datasetImporters;

import java.io.BufferedReader;
import java.io.FileReader;

import weka.core.Instances;

public class ArffImporter {
	public static Instances arffImporter(String path) throws Exception {
		BufferedReader reader = new BufferedReader(new FileReader(path));
		Instances data = new Instances(reader);
		reader.close();
		data.setClassIndex(data.numAttributes() - 1);
		return data;
	}
}
