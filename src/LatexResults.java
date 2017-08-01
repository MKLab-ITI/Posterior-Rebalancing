import java.io.BufferedOutputStream;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.ArrayList;

import org.apache.commons.math3.stat.inference.MannWhitneyUTest;
import org.apache.commons.math3.stat.inference.TTest;

import algorithms.rebalance.ClassRebalance;
import algorithms.rebalance.ExperimentScheme;
import importer.DatasetScheme;
import importer.datasetImporters.DataImporter;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;

public class LatexResults {
	public static ExperimentScheme baseScheme = new ExperimentScheme("Logistic", "", "0");
	public static int experimentFamily = 1;
	public static void main(String[] args) throws Exception {
		int folds = 10;
		
		DataImporter.convertToBinaryDataset = true;
		
		ArrayList<ExperimentScheme> algorithms = new ArrayList<ExperimentScheme>();
		if(experimentFamily==1) {
			algorithms.add(new ExperimentScheme("Logistic", "", "0"));
			algorithms.add(new ExperimentScheme("Logistic", "", "Tinv"));//TIR
			algorithms.add(new ExperimentScheme("Logistic", "", "Tthr"));//TPlugIn
			algorithms.add(new ExperimentScheme("Logistic", "", "Tada"));//RMT
		}
		else if(experimentFamily==2) {
			algorithms.add(new ExperimentScheme("Logistic", "", "0+Sample"));
			algorithms.add(new ExperimentScheme("Logistic", "", "Tinv+Sample"));
			algorithms.add(new ExperimentScheme("Logistic", "", "Tthr+Sample"));
			algorithms.add(new ExperimentScheme("Logistic", "", "Tada+Sample"));
		}
		else if(experimentFamily==3) {
			algorithms.add(new ExperimentScheme("RUSBoost", "", "0"));
			algorithms.add(new ExperimentScheme("RUSBoost", "", "Tinv"));
			algorithms.add(new ExperimentScheme("RUSBoost", "", "Tthr"));
		}
		else if(experimentFamily==4) {
			algorithms.add(new ExperimentScheme("Logistic", "", "0+SMOTE"));
			//algorithms.add(new ExperimentScheme("RUSBoost", "", "0"));
			algorithms.add(new ExperimentScheme("Logistic", "", "Tinv+SMOTE"));
			algorithms.add(new ExperimentScheme("Logistic", "", "Tthr+SMOTE"));
		}
		else if(experimentFamily==5) {
			algorithms.add(new ExperimentScheme("Logistic", "", "0"));
			algorithms.add(new ExperimentScheme("Adaptive", "", "0"));
			algorithms.add(new ExperimentScheme("Logistic", "", "Tthr"));
		}
		ArrayList<DatasetScheme> datasets = new ArrayList<DatasetScheme>();

		datasets.add(new DatasetScheme("Boundary", "data/hddt/imbalanced/boundary.data", ".data", folds));
		datasets.add(new DatasetScheme("Breast-Y", "data/hddt/imbalanced/breast-y.data", ".data", folds));
		datasets.add(new DatasetScheme("Cam", "data/hddt/imbalanced/cam.data", ".data", folds));
		datasets.add(new DatasetScheme("CompuStat", "data/hddt/imbalanced/compustat.data", ".data", folds));
		datasets.add(new DatasetScheme("CovType", "data/hddt/imbalanced/covtype.data", ".data", folds));
		datasets.add(new DatasetScheme("Credit-G", "data/hddt/imbalanced/credit-g.data", ".data", folds));
		datasets.add(new DatasetScheme("Estate", "data/hddt/imbalanced/estate.data", ".data", folds));
		datasets.add(new DatasetScheme("Heart-v", "data/hddt/imbalanced/heart-v.data", ".data", folds));
		datasets.add(new DatasetScheme("Hypo", "data/hddt/imbalanced/hypo.data", ".data", folds));
		datasets.add(new DatasetScheme("ISM", "data/hddt/imbalanced/ism.data", ".data", folds));
		datasets.add(new DatasetScheme("Letter", "data/hddt/imbalanced/letter.data", ".data", folds));
		datasets.add(new DatasetScheme("Oil", "data/hddt/imbalanced/oil.data", ".data", folds));
		datasets.add(new DatasetScheme("Page", "data/hddt/imbalanced/page.data", ".data", folds));
		datasets.add(new DatasetScheme("PenDigits", "data/hddt/imbalanced/pendigits.data", ".data", folds));
		datasets.add(new DatasetScheme("Phoneme", "data/hddt/imbalanced/phoneme.data", ".data", folds));
		datasets.add(new DatasetScheme("PhosS", "data/hddt/imbalanced/PhosS.data", ".data", folds));
		datasets.add(new DatasetScheme("SatImage", "data/hddt/imbalanced/satimage.data", ".data", folds));
		datasets.add(new DatasetScheme("Segment", "data/hddt/imbalanced/segment.data", ".data", folds));
		datasets.add(new DatasetScheme("Sick", "data/hddt/imbalanced/sick.data", ".data", folds));
		
		datasets.add(new DatasetScheme("Adult", "data/UCI/adult.data", ".data", folds));
		datasets.add(new DatasetScheme("Car", "data/UCI/car.data", ".data", folds));
		datasets.add(new DatasetScheme("Contraception", "data/UCI/cmc.data", ".data", folds));
		datasets.add(new DatasetScheme("Glass", "data/UCI/glass.data", ".data", folds));
		datasets.add(new DatasetScheme("Lung Cancer", "data/UCI/lung-cancer.arff", ".arff", folds));
		datasets.add(new DatasetScheme("Page Blocks", "data/UCI/page-blocks.data", ".data", folds));
		datasets.add(new DatasetScheme("Sick Euthyroid", "data/UCI/sick-euthyroid.data", ".data", folds));
		datasets.add(new DatasetScheme("Thoratic Surgery", "data/UCI/ThoraricSurgery.arff", ".arff", folds));
		datasets.add(new DatasetScheme("Yeast", "data/UCI/yeast.data", ".data", folds));
		
		for(DatasetScheme dataset : datasets) {
			System.out.println(dataset.toString()+" imabalance: "+ dataset.measureImbalance());
		}
		
		//System.setOut(new PrintStream(new BufferedOutputStream(new FileOutputStream("out\\"+scheme+".txt")), true));
		
		//System.out.println(scheme+"\n");
		for(DatasetScheme dataset : datasets) {
			System.out.print("\\textbf{"+dataset.toString()+"}");
			for(ExperimentScheme algorithm : algorithms) 
				System.out.print(" & "+latexResults(dataset, algorithm));
			System.out.println("\\\\");
		}
	}
	
	public static String latexResults(DatasetScheme dataset, ExperimentScheme algorithm) {
		String ret = "";
		try{
		Instances instances = dataset.getAllTestInstances();
	    Evaluation eval = new Evaluation(instances);
	    double gain = 0;
	    int gainN = 0;
	    double entropy = 0;

		ArrayList<Double> ranksCorrect = new ArrayList<Double>();
		ArrayList<Double> ranksIncorrect = new ArrayList<Double>();
		ArrayList<Double> entropiesBase = new ArrayList<Double>();
		ArrayList<Double> entropies = new ArrayList<Double>();
		
    	for (int n = 0; n < dataset.getFolds(); n++) {
    		ClassRebalance classifier = (ClassRebalance)algorithm.produceClassifier();
    		ClassRebalance baseClassifier = (ClassRebalance)baseScheme.produceClassifier();
			dataset.produceNextSets(n);
			Instances train = dataset.getTrainSet();
			Instances test = dataset.getTestSet();
			baseClassifier.buildClassifier(train);
			classifier.buildClassifier(train);
			eval.evaluateModel(classifier, test);
			
			for(int i=0;i<test.numInstances();i++) {
				double instanceEntropy = ClassRebalance.normalizedEntropy(classifier.distributionForInstance(test.instance(i)));
				double instanceGain = instanceEntropy
						            - ClassRebalance.normalizedEntropy(baseClassifier.distributionForInstance(test.instance(i)));
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
		ret += toPercentage(GTPr, 0)+" & ";//GM
		ret += toPercentage(fairNom/fairDenom, 0)+" & ";//imbalance
		ret += toPercentage(eval.weightedAreaUnderROC(), 0)+" & ";//wAUC
		ret += toPercentage(gain/gainN,0)+" ";//information loss
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

	private static String toPercentage(double val, int spaces) {
		//return (""+Math.round(val*Math.pow(10, spaces+2))/Math.pow(10, spaces)/100).replace("0.",".");
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
