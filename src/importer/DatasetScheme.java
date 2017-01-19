package importer;
import java.util.Random;

import misc.DimensionReduction;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.StringToNominal;

/**
 * <h1>DatasetScheme</h1>
 * This class is used to import a variety of databases and then generate training and evaluation sub-datasets using
 * either Weka or heuristic cross-validation algorithms. For the purposes of results always being comparable, 
 * a static <code>randomSeed</code> field is introduced, which is always assigned a default value. This field is used
 * to randomize our dataset.<br/>
 * This class supports a lot of utilities, such as using probabilistic mapping between features of different training 
 * and evaluation datasets.<br/>
 * <b>To only train a classifier without evaluation, consider using {@link DatasetScheme#importDataset(String, String)}
 * to obtain required data.</b>
 * 
 * @author Krasanakis Emmanouil
 */
public class DatasetScheme {
	public long randomSeed = 1;//used to randomize the dataset.
	protected double heuristicCrossValidiationSplit = 0;//when 0, use weka instead
	protected int folds;
	protected Instances instances;
	protected Instances trainInstances;
	public static boolean debug = false;
	private Instances nextTrainSet;
	private Instances nextTestSet;
	private String name;
	

	/**
	 * Calls the {@link #DatasetScheme(String, String, String, String, int, double)} constructor using the most
	 * common settings for simple evaluation tasks.
	 * @param name a custom dataset name
	 * @param attribute class attribute name
	 * @param testDataset dataset name or path, as required by {@link #importDataset(String, String)}
	 * @param folds the number of cross-validation folds
	 * @throws Exception
	 */
	public DatasetScheme(String name, String attribute, String dataset, int folds) throws Exception {
		this(attribute, dataset, attribute, dataset, folds, 0);
		this.name = name;
	}
	
	/**
	 * Calls the {@link #DatasetScheme(String, String, String, String, int, double)} constructor using the most
	 * common settings for simple evaluation tasks.
	 * @param name custom dataset name
	 * @param attribute class attribute name
	 * @param testDataset dataset name or path, as required by {@link #importDataset(String, String)}
	 * @param folds the number of cross-validation folds
	 * @throws Exception
	 */
	public DatasetScheme(String attribute, String dataset, int folds) throws Exception {
		this(attribute, dataset, attribute, dataset, folds, 0);
	}
	/**
	 * This constructor is used to generate complex dataset schemes.
	 * @param attribute class attribute name for the evaluation dataset
	 * @param testDataset the evaluation dataset's name or path, as required by {@link #importDataset(String, String)}
	 * @param trainAttribute the class attribute name for the training dataset
	 * @param trainDataset the training dataset's name or path, as required by {@link #importDataset(String, String)}. 
	 * 			If not the same as evaluation dataset, {@link Compliance} is used to forcefully match the datasets. 
	 * @param folds the number of cross-validation folds
	 * @param heuristicCrossValidiationSplit 0 uses Weka cross validation, otherwise performs heuristic cross-validation split
	 * 			using the {@link 
	 * @throws Exception
	 */
	public DatasetScheme(String attribute, String testDataset, String trainAttribute, String trainDataset, int folds, double heuristicCrossValidiationSplit) throws Exception {
		this.folds = folds;
		this.heuristicCrossValidiationSplit = heuristicCrossValidiationSplit;
		if(trainDataset.isEmpty())
			trainDataset = testDataset;
		if(trainAttribute.isEmpty())
			trainAttribute = attribute;
		if(debug) {
			System.out.println("Generating dataset schema for "+folds+" folds:");
			System.out.print("\tImporting instances ("+attribute+" @ "+testDataset+") .. ");
		}
		instances = importDataset(attribute, testDataset);
		if(debug)
			System.out.println("FIN ("+instances.numInstances()+" instances, "+(instances.numAttributes()-1)+" features)");

		if(!trainDataset.equals(testDataset) || !trainAttribute.equals(attribute)) {
			if(debug)
				System.out.print("\tImporting train instances ("+trainAttribute+" @ "+trainDataset+") .. ");
			if(folds!=1) 
				throw new Exception("Must have only a single fold when producing dataset correlation");
			trainInstances = importDataset(trainAttribute, trainDataset);
			if(debug)
				System.out.println("FIN");
		}
		else
			trainInstances = instances;
		
		if(instances!=trainInstances) {
			if(debug)
				System.out.print("\tCreating compliance .. ");
			instances = DimensionReduction.reduce(instances, 1000);
			trainInstances = DimensionReduction.reduce(trainInstances, 1000);
			
			//Transformer transformer = new Transformer();
			//transformer.split(instances, 0.8);
			trainInstances = Compliance.compliantInstances(trainInstances, instances);//transformer.getTraining());
			if(debug)
				System.out.println("FIN");
		}
		instances.randomize(new Random(randomSeed));
		if(heuristicCrossValidiationSplit!=0 && folds>1)
			instances.stratify(folds);
		
		name = getAllTrainInstances().classAttribute().name().replace("/", "_").replace("\\", "_");
	}
	/**
	 * <h1>getFolds</h1>
	 * @return the number of folds specifid for this {@link DatasetScheme}
	 */
	public int getFolds() {
		return folds;
	}
	/**
	 * <h1>getAllTrainInstances</h1>
	 * @return the whole set out of which training instances are extracted by{@link #produceNextSets},
	 * as a {@link weka.core.Instances} object
	 */
	public Instances getAllTrainInstances() {
		return trainInstances;
	}
	/**
	 * <h1>getAllTestInstances</h1>
	 * @return the whole set out of which test (i.e. evaluation) instances are extracted by{@link #produceNextSets},
	 * as a {@link weka.core.Instances} object
	 */
	public Instances getAllTestInstances() {
		return instances;
	}
	/**
	 * <h1>importDataset</h1>
	 * This static function is used to import a single dataset.<br/>
	 * <i>myPersonality</i>, <i>USEMP</i> and <i>UCI</i> datasets are heuristically imported using the appropriate class
	 * from the {@link importer.datasetImporters} package. <code>.arff</code> files are directly imported through their path.
	 * @param attribute a substring of the class attribute in the dataset (<code>.arff</code> files are pressumed to have their last attribute being their class instead)
	 * @param dataset either the path of an <code>.arff</code> file or "myPersonality", "USEMP250" or "UCI"
	 * @return
	 * @throws Exception
	 */
	public static Instances importDataset(String attribute, String dataset) throws Exception {
		Instances instances = null;
		if(attribute.toLowerCase().endsWith(".arff")) 
			instances = importer.datasetImporters.ArffImporter.arffImporter(attribute);
		else if(dataset.toLowerCase().contains("mypersonality"))  
			instances = importer.datasetImporters.MyPersonalityImporter.importLDA("data/myPersonality/"+attribute+"/");
		else if(dataset.toLowerCase().contains("usemp250")) {
			attribute = attribute.toLowerCase();
			instances = importer.datasetImporters.ArffImporter.arffImporter("data/usemp250/LDA_ml_100_glo-L2.arff");
			//keep only needed attributes
			Remove remove = new Remove();
			String removeRangeList = "";
			for(int i=0;i<instances.numAttributes();i++)
				if(!instances.attribute(i).toString().startsWith("LDA") && instances.attribute(i).toString().toLowerCase().contains(attribute)) {
					if(!removeRangeList.isEmpty())
						removeRangeList += ",";
					removeRangeList += i;
				}
			remove.setAttributeIndices(removeRangeList);
			remove.setInputFormat(instances);
			instances = Filter.useFilter(instances, remove);
			//convert string attributes to nominal attributes
			String stringAttributes = "";
			for(int i=0;i<instances.numAttributes();i++)
				if(!instances.attribute(i).isString()) {
					if(!stringAttributes.isEmpty())
						stringAttributes += ",";
					stringAttributes += i;
				} 
			if(!stringAttributes.isEmpty()) {
				StringToNominal stringToNominal = new StringToNominal();
				stringToNominal.setAttributeRange(stringAttributes);
				stringToNominal.setInputFormat(instances);
				instances = Filter.useFilter(instances, stringToNominal);
			}
			int classAttr = -1;
			for(int i=0;i<instances.numAttributes();i++)
				if(instances.attribute(i).toString().toLowerCase().contains(attribute))
					classAttr = i;
			if(classAttr==-1)
				throw new Exception("Could not find attribute "+attribute+" in dataset");
			instances.setClassIndex(classAttr);
		}
		else if(dataset.toLowerCase().contains("usemp"))
			instances = importer.datasetImporters.USEMPImporter.importDatabase("data/usemp/", attribute);
		else if(dataset.contains("UCI")) {
			if(attribute.contains("spam")) {
				String[] classes = {"ham", "spam"};
				instances = importer.datasetImporters.UCIImporter.importDatabase("data/UCI/smsspamcollection/SMSSpamCollection", classes, 0);
			}
			else if(attribute.contains("cell")) {
				String[] classes = {"0", "1"};
				instances = importer.datasetImporters.UCIImporter.importDatabase("data/UCI/sentiment labelled sentences/amazon_cells_labelled.txt", classes, -1);
			}
		}
		if(instances==null)
			throw new Exception("Dataset "+dataset+" does not support attribute: "+attribute);
		{
			int i = 0;
			while(i<instances.numInstances()) {
				if(instances.instance(i).classIsMissing())
					instances.delete(i);
				else
					i++;
			}
		}
		return instances;
	}
	
	/**
	 * <h1>produceNextSets</h1>
	 * Produces the next training and test (i.e. evaluation) datasets used for
	 * cross-validation.
	 * @param fold the current fold for cross-validation (some setups don't require this parameter)
	 * @throws Exception
	 * @see #getTrainSet()
	 * @see #getTestSet()
	 */
	public void produceNextSets(int fold) throws Exception {
		if(folds<=1 && trainInstances!=instances) {
			nextTrainSet = trainInstances;
			nextTestSet = instances;
		}
		else if(heuristicCrossValidiationSplit!=0) {
			instances.randomize(new Random((long)(1000*Math.random())));
			DatasetSplitter transformer = new DatasetSplitter();
			transformer.split(instances, heuristicCrossValidiationSplit);
			nextTrainSet = transformer.getTraining();
			nextTestSet = transformer.getTest();
		}
		else if (folds>1) {
			nextTrainSet = instances.trainCV(folds, fold);
			nextTestSet = instances.testCV(folds, fold);
		}
		else {
			nextTrainSet = trainInstances;
			nextTestSet = instances;
		}
	}
	/**
	 * <h1>getTrainSet</h1>
	 * @return the training set last produced by {@link #produceNextSets} as a {@link weka.core.Instances} object
	 * @throws Exception
	 */
	public Instances getTrainSet() throws Exception {
		if(nextTrainSet==null)
			throw new Exception("Call produceNextSets before getTrainSet");
		return nextTrainSet;
	}
	/**
	 * <h1>getTestSet</h1>
	 * @return the test set last produced by {@link #produceNextSets} as a {@link weka.core.Instances} object
	 * @throws Exception
	 */
	public Instances getTestSet() throws Exception {
		if(nextTrainSet==null)
			throw new Exception("Call produceNextSets before getTestSet");
		return nextTestSet;
	}
	
	@Override
	public String toString() {
		return name;
	}
}
