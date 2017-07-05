import java.io.BufferedOutputStream;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.ArrayList;

import algorithms.rebalance.ClassRebalance;
import algorithms.rebalance.ExperimentScheme;
import importer.DatasetScheme;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;

public class LatexResults {
	public static ExperimentScheme baseScheme = new ExperimentScheme("Logistic", "", "0");
	public static int experimentFamily = 3;
	public static void main(String[] args) throws Exception {
		int folds = 10;
		
		ArrayList<ExperimentScheme> algorithms = new ArrayList<ExperimentScheme>();
		if(experimentFamily==1) {
			algorithms.add(new ExperimentScheme("Logistic", "", "0"));
			algorithms.add(new ExperimentScheme("Logistic", "", "Tinv"));
			algorithms.add(new ExperimentScheme("Logistic", "", "Tthr"));
		}
		else if(experimentFamily==2) {
			algorithms.add(new ExperimentScheme("Logistic", "", "0+Sample"));
			//algorithms.add(new ExperimentScheme("RUSBoost", "", "0"));
			algorithms.add(new ExperimentScheme("Logistic", "", "Tinv+Sample"));
			algorithms.add(new ExperimentScheme("Logistic", "", "Tthr+Sample"));
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
		ArrayList<DatasetScheme> datasets = new ArrayList<DatasetScheme>();
		datasets.add(new DatasetScheme("Adult", "data/UCI/adult.data", ".data", folds));
		datasets.add(new DatasetScheme("Car", "data/UCI/car.data", ".data", folds));
		datasets.add(new DatasetScheme("Contraception", "data/UCI/cmc.data", ".data", folds));
		datasets.add(new DatasetScheme("Glass", "data/UCI/glass.data", ".data", folds));
		datasets.add(new DatasetScheme("Lung Cancer", "data/UCI/Medical/lung-cancer.arff", ".arff", folds));
		datasets.add(new DatasetScheme("Page Blocks", "data/UCI/page-blocks.data", ".data", folds));
		datasets.add(new DatasetScheme("Sick Euthyroid", "data/UCI/sick-euthyroid.data", ".data", folds));
		datasets.add(new DatasetScheme("Thoratic Surgery", "data/UCI/ThoraricSurgery.arff", ".arff", folds));
		datasets.add(new DatasetScheme("Yeast", "data/UCI/yeast.data", ".data", folds));
		
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
					gain -= instanceGain;
				else if((classifier.classifyInstance(test.instance(i))!=test.instance(i).classValue()) 
						&& (baseClassifier.classifyInstance(test.instance(i))==test.instance(i).classValue()))
					gain += instanceGain;
				else if((classifier.classifyInstance(test.instance(i))!=test.instance(i).classValue()) 
						&& (baseClassifier.classifyInstance(test.instance(i))!=test.instance(i).classValue()))
					gain -= instanceGain;
				if(classifier.classifyInstance(test.instance(i))==test.instance(i).classValue())
					entropy += instanceEntropy;
				//else
					//entropy -= instanceEntropy;
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
	
		//ret += toPercentage(eval.weightedTruePositiveRate(), 0)+"\\% & ";//wTPr
		ret += toPercentage(GTPr, 0)+" & ";//GM
		ret += toPercentage(fairNom/fairDenom, 0)+" & ";//imbalance
		ret += toPercentage(eval.weightedAreaUnderROC(), 0)+" & ";//wAUC
		ret += toPercentage(gain/gainN,0)+"b ";//information loss
		}
		catch(Exception e) {
			System.err.println(e);
			ret = "";
		}
		return ret;
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
