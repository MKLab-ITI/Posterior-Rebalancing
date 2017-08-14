import java.util.ArrayList;

import org.apache.commons.math3.stat.inference.MannWhitneyUTest;
import org.apache.commons.math3.stat.inference.TTest;

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
		
		// report dataset balance
		for(DatasetScheme dataset : datasetScemes) 
			System.out.println(dataset.toString()+" imabalance: "+ dataset.measureImbalance());
		
		// set output to file
		//System.setOut(new PrintStream(new BufferedOutputStream(new FileOutputStream("out\\"+scheme+".txt")), true));
		
		String tableSeparator = " & ";
		for(DatasetScheme dataset : datasetScemes) {
			System.out.print("\\textbf{"+dataset.toString()+"}");
			for(ExperimentScheme experimentScheme : experimentSchemes) 
				System.out.print(tableSeparator+latexResults(tableSeparator, dataset, experimentScheme, baseExperimentScheme));
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
	public static String latexResults(String separator, DatasetScheme datasetScheme, ExperimentScheme experimentScheme, ExperimentScheme baseExperimentScheme) {
		String ret = "";
		try{
		Instances instances = datasetScheme.getAllTestInstances();
	    Evaluation eval = new Evaluation(instances);
	    double gain = 0;
	    int gainN = 0;
	    double entropy = 0;
		ArrayList<Double> ranksCorrect = new ArrayList<Double>();
		ArrayList<Double> ranksIncorrect = new ArrayList<Double>();
		ArrayList<Double> entropiesBase = new ArrayList<Double>();
		ArrayList<Double> entropies = new ArrayList<Double>();
		
    	for (int n = 0; n < datasetScheme.getFolds(); n++) {
    		ClassRebalance classifier = (ClassRebalance)experimentScheme.produceClassifier();
    		ClassRebalance baseClassifier = (ClassRebalance)baseExperimentScheme.produceClassifier();
			datasetScheme.produceNextSets(n);
			Instances train = datasetScheme.getTrainSet();
			Instances test = datasetScheme.getTestSet();
			baseClassifier.buildClassifier(train);
			classifier.buildClassifier(train);
			eval.evaluateModel(classifier, test);
			
			for(int i=0;i<test.numInstances();i++) {
				double instanceEntropy = ClassRebalance.normalizedEntropy(classifier.distributionForInstance(test.instance(i)));
				double instanceGain = instanceEntropy - ClassRebalance.normalizedEntropy(baseClassifier.distributionForInstance(test.instance(i)));
				if((classifier.classifyInstance(test.instance(i))==test.instance(i).classValue()) 
						&& (baseClassifier.classifyInstance(test.instance(i))==test.instance(i).classValue()))
					gain += instanceGain;
				else if((classifier.classifyInstance(test.instance(i))==test.instance(i).classValue()) 
						&& (baseClassifier.classifyInstance(test.instance(i))!=test.instance(i).classValue()))
					{}//gain -= instanceGain;
				else if((classifier.classifyInstance(test.instance(i))!=test.instance(i).classValue()) 
						&& (baseClassifier.classifyInstance(test.instance(i))==test.instance(i).classValue()))
					gain += instanceGain;
				else if((classifier.classifyInstance(test.instance(i))!=test.instance(i).classValue()) 
						&& (baseClassifier.classifyInstance(test.instance(i))!=test.instance(i).classValue()))
					{}//gain -= instanceGain;
				if(classifier.classifyInstance(test.instance(i))==test.instance(i).classValue())
					entropy += instanceEntropy;
				//else
					//entropy -= instanceEntropy;
				
				int classifyInstance = (int) baseClassifier.classifyInstance(test.instance(i));
				if(classifyInstance==test.instance(i).classValue())
					ranksCorrect.add(-instanceGain);
				else
					ranksIncorrect.add(-instanceGain);
				
				if(baseClassifier.classifyInstance(test.instance(i))==test.instance(i).classValue())
					entropiesBase.add(instanceEntropy-instanceGain);
				else
					entropiesBase.add(-instanceEntropy+instanceGain);

				if(classifier.classifyInstance(test.instance(i))==test.instance(i).classValue())
					entropies.add(instanceEntropy);
				else
					entropies.add(-instanceEntropy);
				
				gainN++;
			}
		}
	    
	    //eval.crossValidateModel(algorithm.produceClassifier(), instances, dataset.getFolds(), new java.util.Random(1));
		
	    double[] classFrequencies = eval.getClassPriors();
		double sumFrequencies = 0;
		for(int i=0;i<classFrequencies.length;i++)
			sumFrequencies += classFrequencies[i];
		for(int i=0;i<classFrequencies.length;i++)
			classFrequencies[i] /= sumFrequencies;
		double meanTPr = 0;
		double GTPr = 1;
		int count = 0;
	    for(int i=0;i<instances.numClasses();i++)  {
	    	if(classFrequencies[i]==0)
	    		continue;
		    meanTPr += eval.truePositiveRate(i)/instances.numClasses();
		    count++;
		    GTPr *= eval.truePositiveRate(i);
	    }
	    GTPr =  Math.pow(GTPr,1.0/count);
	    
	    double fairNom = 0;
		double fairDenom = 0;
		for(int i=0;i<instances.numClasses();i++) {
			if(classFrequencies[i]==0)
				continue;
		    for(int j=0;j<instances.numClasses();j++)
			    if(i!=j){
			    	if(classFrequencies[j]==0)
			    		continue;
			    	fairNom += classFrequencies[i]*classFrequencies[j]*Math.abs(eval.truePositiveRate(i)-eval.truePositiveRate(j));
			    	fairDenom += classFrequencies[i]*classFrequencies[j];
			    }
		}
		
		MannWhitneyUTest uTest = new MannWhitneyUTest();
		double U = (double)uTest.mannWhitneyU(toPrimitive(ranksCorrect), toPrimitive(ranksIncorrect));
		U = U/ranksCorrect.size()/ranksIncorrect.size();
		TTest tTest = new TTest();
		double T = (double)tTest.tTest(toPrimitive(entropies), toPrimitive(entropiesBase));
	
		//ret += toPercentage(eval.weightedTruePositiveRate(), 0)+"\\% & ";//wTPr
		ret += toPercentage(GTPr)+separator;//GM
		ret += toPercentage(fairNom/fairDenom)+separator;//imbalance
		ret += toPercentage(eval.weightedAreaUnderROC())+separator;//wAUC
		ret += toPercentage(gain/gainN);//information loss
		//ret += toPercentage(U,0)+"U ";//measures whether distributions differ between correct and incorrect classifications (0.5=don't differ, 1=certain)
		//ret += toPercentage(T,0)+"T ";
		//ret += toPercentage(2*U-1,0)+"r ";//rank-biserial correlation, i.e. the proportion of samples that confirm the hypothesis that distributions differ between correct and incorrect
		}
		catch(Exception e) {
			System.err.println(e);
			ret = "";
		}
		return ret;
	}
	
	/**
	 * <h1>toPrimitive</h1>
	 * @param array an array list of doubles
	 * @return an array representation of the give array list
	 */
	public static double[] toPrimitive(ArrayList<Double> array) {
		  if (array == null) {
		    return null;
		  } else if (array.size() == 0) {
		    return new double[0];
		  }
		  final double[] result = new double[array.size()];
		  for (int i = 0; i < array.size(); i++) {
		    result[i] = array.get(i).doubleValue();
		  }
		  return result;
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
