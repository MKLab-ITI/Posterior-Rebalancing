import java.io.BufferedOutputStream;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.ArrayList;

import algorithms.rebalance.ExperimentScheme;
import importer.DatasetScheme;
import weka.classifiers.Evaluation;
import weka.core.Instances;

public class LatexResults {

	public static void main(String[] args) throws Exception {
		int folds = 10;
		
		String scheme = "UTinv+smote";
		
		ArrayList<ExperimentScheme> algorithms = new ArrayList<ExperimentScheme>();
		algorithms.add(new ExperimentScheme("Logistic", "", scheme));
		algorithms.add(new ExperimentScheme("SKNN", "", scheme));
		algorithms.add(new ExperimentScheme("SVM", "", scheme));
		
		ArrayList<DatasetScheme> datasets = new ArrayList<DatasetScheme>();
		datasets.add(new DatasetScheme("Sexuality \\#1", "sexuality", "myPersonality", 2));
		datasets.add(new DatasetScheme("Religion \\#1", "religion", "myPersonality", folds));
		datasets.add(new DatasetScheme("Politics \\#1", "politics", "myPersonality", folds));
		
		datasets.add(new DatasetScheme("Sexuality \\#2", "sex", "usemp250", folds));
		datasets.add(new DatasetScheme("Religion \\#2", "relig", "usemp250", folds));
		datasets.add(new DatasetScheme("Politics \\#2", "polit", "usemp250", folds));
		datasets.add(new DatasetScheme("Cannabis User", "cannabis", "usemp250", folds));
		datasets.add(new DatasetScheme("Spam Collection", "spam", "UCI", 2));
		datasets.add(new DatasetScheme("Cell Review Sentiment", "cell", "UCI", 2));
		datasets.add(new DatasetScheme("Dermatology", "data/UCI/Medical/dermatology.arff", ".arff", folds));
		datasets.add(new DatasetScheme("Diabetes", "data/UCI/Medical/diabetes.arff", ".arff", folds));
		datasets.add(new DatasetScheme("E. Coli", "data/UCI/Medical/ecoli.arff", ".arff", folds));
		datasets.add(new DatasetScheme("Hepatitis", "data/UCI/Medical/hepatitis.arff", ".arff", folds));
		datasets.add(new DatasetScheme("Hypothyroid", "data/UCI/Medical/hypothyroid.arff", ".arff", folds));
		datasets.add(new DatasetScheme("Liver Disorders", "data/UCI/Medical/liver-disorders.arff", ".arff", folds));
		datasets.add(new DatasetScheme("Lung Cancer", "data/UCI/Medical/lung-cancer.arff", ".arff", folds));
		datasets.add(new DatasetScheme("Lymphography", "data/UCI/Medical/lymph.arff", ".arff", folds));
		
		System.setOut(new PrintStream(new BufferedOutputStream(new FileOutputStream("out\\"+scheme+".txt")), true));
		
		System.out.println(scheme+"\n");
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
	    eval.crossValidateModel(algorithm.produceClassifier(), instances, dataset.getFolds(), new java.util.Random(1));
		
		double[] classFrequencies = eval.getClassPriors();
		double sumFrequencies = 0;
		for(int i=0;i<classFrequencies.length;i++)
			sumFrequencies += classFrequencies[i];
		for(int i=0;i<classFrequencies.length;i++)
			classFrequencies[i] /= sumFrequencies;
	    
		double fairNom = 0;
		double fairDenom = 0;
	    for(int i=0;i<instances.numClasses();i++) 
		    for(int j=0;j<instances.numClasses();j++) {
		    	double weight = (classFrequencies[i]*classFrequencies[j]);
		    	if(i!=j && eval.truePositiveRate(i)!=0 && eval.truePositiveRate(j)!=0) {
		    		double coupleUnfairness = eval.truePositiveRate(i)/eval.truePositiveRate(j)+eval.truePositiveRate(j)/eval.truePositiveRate(i);
		    		fairNom += 2*weight/coupleUnfairness;
		    	}
		    	if(i!=j)
		    		fairDenom += weight;
		    }
	    
		ret += toPercentage(eval.weightedAreaUnderROC(), 0)+"\\% & ";//wAUC
		ret += toPercentage(eval.weightedTruePositiveRate(), 0)+"\\% & ";//wTPr
		ret += toPercentage(fairNom/fairDenom, 0)+"\\% & ";//fairness
		ret += toPercentage(fairNom/fairDenom*eval.weightedTruePositiveRate(), 0)+"\\%";
		}
		catch(Exception e) {
			System.err.println(e);
			ret = "";
		}
		return ret;
	}

	private static String toPercentage(double val, int spaces) {
		if(spaces==0)
			return ""+Math.round(val*100);
		return ""+Math.round(val*Math.pow(10, spaces+2))/Math.pow(10, spaces);
	}
}
