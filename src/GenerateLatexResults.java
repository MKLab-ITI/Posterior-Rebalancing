import java.util.ArrayList;

import algorithms.rebalance.ClassRebalance;
import algorithms.rebalance.ExperimentScheme;
import importer.DatasetScheme;
import weka.classifiers.Evaluation;
import weka.core.Instances;

public class GenerateLatexResults {
	public static enum ExperimentFamily {postProcess, postProcessSampling, postProcessSMOTE};

	public static void main(String[] args) throws Exception {
		// load datasets and generate experiment schemes
		ArrayList<DatasetScheme> datasetScemes = importDatasetSchemes(10, true);//load 10-fold cross validation datasets
		ArrayList<ExperimentScheme> experimentSchemes = generateExperimentSchemes("Logistic", ExperimentFamily.postProcess);
		//extract base scheme from experiment schemes
		ExperimentScheme baseExperimentScheme = experimentSchemes.get(0);
		experimentSchemes.remove(baseExperimentScheme);
		//add metrics to experiment with
		ArrayList<EvaluationMetric> metrics = new ArrayList<EvaluationMetric>();
		metrics.add(new EvaluationMetric.GM());
		metrics.add(new EvaluationMetric.Imbalance());
		metrics.add(new EvaluationMetric.AUC());
		metrics.add(new EvaluationMetric.ILoss());
		
		// set output to file
		//System.setOut(new PrintStream(new BufferedOutputStream(new FileOutputStream("out\\"+scheme+".txt")), true));
		
		String tableSeparator = " & ";
		for(DatasetScheme dataset : datasetScemes) {
			System.out.print("\\textbf{"+dataset.toString()+"}");
			for(ExperimentScheme experimentScheme : experimentSchemes) 
				System.out.print(tableSeparator+latexResults(tableSeparator, dataset, experimentScheme, baseExperimentScheme, metrics));
			System.out.println("\\\\");
		}
		
		// notify user that experiments finished
		java.awt.Toolkit.getDefaultToolkit().beep();
	}
	
	
	/** 
	 * <h1>importDatasetSchemes</h1>
	 * @param evaluationFolds number of folds used for cross-validation for each dataset
	 * @return loads a number of imbalanced datasets
	 * @throws Exception
	 */
	public static ArrayList<DatasetScheme> importDatasetSchemes(int evaluationFolds, boolean convertToBinaryDataset) throws Exception {
		ArrayList<DatasetScheme> datasetSchemes = new ArrayList<DatasetScheme>();
		//HDDT datasets
		datasetSchemes.add(new DatasetScheme("Boundary", "data/hddt/imbalanced/boundary.data", ".data", evaluationFolds, convertToBinaryDataset));
		datasetSchemes.add(new DatasetScheme("Breast-Y", "data/hddt/imbalanced/breast-y.data", ".data", evaluationFolds, convertToBinaryDataset));
		datasetSchemes.add(new DatasetScheme("Cam", "data/hddt/imbalanced/cam.data", ".data", evaluationFolds, convertToBinaryDataset));
		datasetSchemes.add(new DatasetScheme("CompuStat", "data/hddt/imbalanced/compustat.data", ".data", evaluationFolds, convertToBinaryDataset));
		datasetSchemes.add(new DatasetScheme("CovType", "data/hddt/imbalanced/covtype.data", ".data", evaluationFolds, convertToBinaryDataset));
		datasetSchemes.add(new DatasetScheme("Credit-G", "data/hddt/imbalanced/credit-g.data", ".data", evaluationFolds, convertToBinaryDataset));
		datasetSchemes.add(new DatasetScheme("Estate", "data/hddt/imbalanced/estate.data", ".data", evaluationFolds, convertToBinaryDataset));
		datasetSchemes.add(new DatasetScheme("Heart-v", "data/hddt/imbalanced/heart-v.data", ".data", evaluationFolds, convertToBinaryDataset));
		datasetSchemes.add(new DatasetScheme("Hypo", "data/hddt/imbalanced/hypo.data", ".data", evaluationFolds, convertToBinaryDataset));
		datasetSchemes.add(new DatasetScheme("ISM", "data/hddt/imbalanced/ism.data", ".data", evaluationFolds, convertToBinaryDataset));
		datasetSchemes.add(new DatasetScheme("Letter", "data/hddt/imbalanced/letter.data", ".data", evaluationFolds, convertToBinaryDataset));
		datasetSchemes.add(new DatasetScheme("Oil", "data/hddt/imbalanced/oil.data", ".data", evaluationFolds, convertToBinaryDataset));
		datasetSchemes.add(new DatasetScheme("Page", "data/hddt/imbalanced/page.data", ".data", evaluationFolds, convertToBinaryDataset));
		datasetSchemes.add(new DatasetScheme("PenDigits", "data/hddt/imbalanced/pendigits.data", ".data", evaluationFolds, convertToBinaryDataset));
		datasetSchemes.add(new DatasetScheme("Phoneme", "data/hddt/imbalanced/phoneme.data", ".data", evaluationFolds, convertToBinaryDataset));
		datasetSchemes.add(new DatasetScheme("PhosS", "data/hddt/imbalanced/PhosS.data", ".data", evaluationFolds, convertToBinaryDataset));
		datasetSchemes.add(new DatasetScheme("SatImage", "data/hddt/imbalanced/satimage.data", ".data", evaluationFolds, convertToBinaryDataset));
		datasetSchemes.add(new DatasetScheme("Segment", "data/hddt/imbalanced/segment.data", ".data", evaluationFolds, convertToBinaryDataset));
		datasetSchemes.add(new DatasetScheme("Sick", "data/hddt/imbalanced/sick.data", ".data", evaluationFolds, convertToBinaryDataset));
		//UCI datasets
		datasetSchemes.add(new DatasetScheme("Adult", "data/UCI/adult.data", ".data", evaluationFolds, convertToBinaryDataset));
		datasetSchemes.add(new DatasetScheme("Car", "data/UCI/car.data", ".data", evaluationFolds, convertToBinaryDataset));
		datasetSchemes.add(new DatasetScheme("Contraception", "data/UCI/cmc.data", ".data", evaluationFolds, convertToBinaryDataset));
		datasetSchemes.add(new DatasetScheme("Glass", "data/UCI/glass.data", ".data", evaluationFolds, convertToBinaryDataset));
		datasetSchemes.add(new DatasetScheme("Lung Cancer", "data/UCI/lung-cancer.arff", ".arff", evaluationFolds, convertToBinaryDataset));
		datasetSchemes.add(new DatasetScheme("Page Blocks", "data/UCI/page-blocks.data", ".data", evaluationFolds, convertToBinaryDataset));
		datasetSchemes.add(new DatasetScheme("Sick Euthyroid", "data/UCI/sick-euthyroid.data", ".data", evaluationFolds, convertToBinaryDataset));
		datasetSchemes.add(new DatasetScheme("Thoratic Surgery", "data/UCI/ThoraricSurgery.arff", ".arff", evaluationFolds, convertToBinaryDataset));
		datasetSchemes.add(new DatasetScheme("Yeast", "data/UCI/yeast.data", ".data", evaluationFolds, convertToBinaryDataset));
		//return list of schemes
		return datasetSchemes;
	}
	
	public static ArrayList<ExperimentScheme> generateExperimentSchemes(String baseClassifier, ExperimentFamily experimentFamily) {
		ArrayList<ExperimentScheme> experimentSchemes = new ArrayList<ExperimentScheme>();
		if(experimentFamily==ExperimentFamily.postProcess) {
			experimentSchemes.add(new ExperimentScheme(baseClassifier, "", "0"));
			experimentSchemes.add(new ExperimentScheme(baseClassifier, "", "0"));
			experimentSchemes.add(new ExperimentScheme(baseClassifier, "", "Tinv"));//TIR
			experimentSchemes.add(new ExperimentScheme(baseClassifier, "", "Tthr"));//TPlugIn
			experimentSchemes.add(new ExperimentScheme(baseClassifier, "", "Tada"));//RMT
		}
		else if(experimentFamily==ExperimentFamily.postProcessSampling) {
			experimentSchemes.add(new ExperimentScheme(baseClassifier, "", "0"));
			experimentSchemes.add(new ExperimentScheme(baseClassifier, "", "0+Sample"));
			experimentSchemes.add(new ExperimentScheme(baseClassifier, "", "Tinv+Sample"));
			experimentSchemes.add(new ExperimentScheme(baseClassifier, "", "Tthr+Sample"));
			experimentSchemes.add(new ExperimentScheme(baseClassifier, "", "Tada+Sample"));
		}
		else if(experimentFamily==ExperimentFamily.postProcessSMOTE) {
			experimentSchemes.add(new ExperimentScheme(baseClassifier, "", "0"));
			experimentSchemes.add(new ExperimentScheme(baseClassifier, "", "0+SMOTE"));
			experimentSchemes.add(new ExperimentScheme(baseClassifier, "", "Tinv+SMOTE"));
			experimentSchemes.add(new ExperimentScheme(baseClassifier, "", "Tthr+SMOTE"));
		}
		//algorithms.add(new ExperimentScheme("RUSBoost", "", "0"));
		return experimentSchemes;
	}
	
	
	/**
	 * <h1>latexResults</h1>
	 * @param datasetScheme
	 * @param experimentScheme
	 * @return evaluation for given experiment scheme on the given dataset
	 */
	public static String latexResults(String separator, DatasetScheme datasetScheme, ExperimentScheme experimentScheme, ExperimentScheme baseExperimentScheme, ArrayList<EvaluationMetric> metrics) {
		String ret = "";
		for(EvaluationMetric metric : metrics)
			metric.clear();
		try{
			Instances instances = datasetScheme.getAllTestInstances();
			//perform cross-validation evaluation on metrics
		    Evaluation eval = new Evaluation(instances);
	    	for (int n = 0; n < datasetScheme.getFolds(); n++) {
	    		ClassRebalance classifier = (ClassRebalance)experimentScheme.produceClassifier();
	    		ClassRebalance baseClassifier = (ClassRebalance)baseExperimentScheme.produceClassifier();
				datasetScheme.produceNextSets(n);
				Instances train = datasetScheme.getTrainSet();
				Instances test = datasetScheme.getTestSet();
				classifier.buildClassifier(train);
				baseClassifier.buildClassifier(train);
				eval.evaluateModel(classifier, test);
				for(EvaluationMetric metric : metrics)
					metric.validateInstances(test, classifier, baseClassifier);
			}
			//obtain imbalance measures
			for(EvaluationMetric metric : metrics)
				ret += toPercentage(metric.getValue(eval))+metric.unit()+separator;
			ret = ret.substring(0, ret.length()-separator.length());
		}
		catch(Exception e) {
			System.err.println(e);
			ret = "";
		}
		return ret;
	}

	/**
	 * <h1>toPercentage</h1>
	 * @param val a given value
	 * @return a string percentage representation of the given value
	 */
	private static String toPercentage(double val) {
		if(Math.round(val*100)==0)
			return "0";
		if(Math.round(val*100)==100)
			return "1";
		String ret = "";
		if(val<0)
			ret += "-";
		val = Math.abs(val);
		ret += "."+(Math.round(val*100)<10?"0":"")+Math.round(val*100);
		return ret;
	}
}
