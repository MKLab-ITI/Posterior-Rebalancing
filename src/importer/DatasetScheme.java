package importer;
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Random;

import algorithms.rebalance.DatasetMetrics;
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
	protected int folds;
	protected Instances instances;
	private Instances nextTrainSet;
	private Instances nextTestSet;
	private String name;
	protected double baggingSplit;//when 0, use cross-validation instead
	

	public DatasetScheme(String path, String dataType, int folds, boolean convertToBinaryIfPossible) throws Exception {
		this("", path, dataType, folds, 0, true);
	}
	
	public DatasetScheme(String name, String path, String dataType, int folds, boolean convertToBinaryIfPossible) throws Exception {
		this(name, path, dataType, folds, 0, true);
	}

	public DatasetScheme(String name, String path, String dataType, int folds, double baggingSplit, boolean convertToBinaryIfPossible) throws Exception {
		this.folds = folds;
		this.baggingSplit = baggingSplit;
		instances = importDataset(path, dataType, convertToBinaryIfPossible, true);
		this.name = name.isEmpty()?instances.classAttribute().name().replace("/", "_").replace("\\", "_"):name;
		System.out.println(this.name+": "+instances.numInstances()+" instances, "+(instances.numAttributes()-1)+" features, "+instances.numClasses()+" classes");
		
		instances.randomize(new Random(randomSeed));
		if(baggingSplit!=0 && folds>1)
			instances.stratify(folds);
		
	}
	/**
	 * <h1>getFolds</h1>
	 * @return the number of folds specifid for this {@link DatasetScheme}
	 */
	public int getFolds() {
		return folds;
	}
	/**
	 * <h1>getAllTestInstances</h1>
	 * @return the whole set out of which test (i.e. evaluation) instances are extracted by{@link #produceNextSets},
	 * as a {@link weka.core.Instances} object
	 */
	public Instances getAllTestInstances() {
		return instances;
	}
	
	public static Instances importDataset(String path, String dataType, boolean convertToBinaryIfPossible, boolean deleteMissingClassInstances) throws Exception {
		Instances instances = null;
		if(dataType.toLowerCase().contains(".arff")) {
			BufferedReader reader = new BufferedReader(new FileReader(path));
			Instances data = new Instances(reader);
			reader.close();
			data.setClassIndex(data.numAttributes() - 1);
			return data;
		}
		else if(dataType.toLowerCase().contains("mypersonality"))  
			instances = importer.MyPersonalityImporter.importLDA("data/myPersonality/"+path+"/");
		else if(dataType.contains(".data")) {
			instances = importer.DataImporter.importDatabase(path, convertToBinaryIfPossible);
		}
		if(instances==null)
			throw new Exception("Dataset "+dataType+" does not support path or attribute: "+path);
		if(deleteMissingClassInstances)
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
		if(baggingSplit!=0) {
			instances.randomize(new Random((long)(1000*Math.random())));
			DatasetSplitter splitter = new DatasetSplitter();
			splitter.split(instances, baggingSplit);
			nextTrainSet = splitter.getTraining();
			nextTestSet = splitter.getTest();
		}
		else if (folds>1) {
			nextTrainSet = instances.trainCV(folds, fold);
			nextTestSet = instances.testCV(folds, fold);
		}
		else {
			nextTrainSet = instances;
			nextTestSet = instances;
			System.err.println("Parameters yield identical similar and test sets");
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

	public double measureImbalance() {
		double[] frequencies = DatasetMetrics.getPriors(instances);
		if(frequencies.length!=2) {
			double ret = 0;
			for(int i=0;i<frequencies.length;i++)
				for(int j=i+1;j<frequencies.length;j++)
					ret += frequencies[i]/frequencies[j]+frequencies[j]/frequencies[i];
			System.err.println("Heuristic imbalance measure for dataset "+toString());
			return ret*2/frequencies.length/(frequencies.length-1);
		}
		else
			return Math.max(frequencies[0]/frequencies[1], frequencies[1]/frequencies[0]);
	}
}
