
import java.io.File;

import weka.core.Instances;
import weka.core.converters.ArffSaver;

public class PersonalityArrfGenerator {
	public static void main(String[] args) throws Exception {
		Instances trainingInstances = importer.datasetImporters.MyPersonalityImporter.importLDA("data/myPersonality/politics");
		ArffSaver saver = new ArffSaver();
		saver.setInstances(trainingInstances);
		saver.setFile(new File ("myPersonalityPolitics.arff"));
		saver.writeBatch();
	}
}
