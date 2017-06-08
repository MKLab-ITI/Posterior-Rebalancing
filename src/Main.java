import java.io.BufferedOutputStream;
import java.io.FileOutputStream;
import java.io.PrintStream;

import algorithms.rebalance.ExperimentScheme;
import importer.DatasetScheme;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;

public class Main {
	public static void main(String[] args) throws Exception {
		//IMPORT INSTANCES
		ExperimentScheme experimentSchema = new ExperimentScheme("RUSBoost", "", "D1inv");
		experimentSchema.appendRebalanceOptions("-rebalance -0.1");
		int folds = 10;
		//DatasetScheme databaseSchema = new DatasetScheme("sexuality", "myPersonality", folds);
		//DatasetScheme databaseSchema = new DatasetScheme("religion", "myPersonality", folds);
		//DatasetScheme databaseSchema = new DatasetScheme("politics", "myPersonality", folds);
		//DatasetScheme databaseSchema = new DatasetScheme("sex", "usemp250", folds);//don't forget to also make -sensitive 1 or 2 for usemp250
		//DatasetScheme databaseSchema = new DatasetScheme("relig", "usemp250", folds);
		//DatasetScheme databaseSchema = new DatasetScheme("polit", "usemp250", folds);
		//DatasetScheme databaseSchema = new DatasetScheme("cannabis", "usemp250", folds);
		//DatasetScheme databaseSchema = new DatasetScheme("spam", "UCI", folds);
		//DatasetScheme databaseSchema = new DatasetScheme("cell", "UCI", folds);
		//DatasetScheme databaseSchema = new DatasetScheme("data/UCI/Medical/dermatology.arff", ".arff", folds);
		//DatasetScheme databaseSchema = new DatasetScheme("data/UCI/Medical/diabetes.arff", ".arff", folds);
		//DatasetScheme databaseSchema = new DatasetScheme("data/UCI/Medical/ecoli.arff", ".arff", folds);
		//DatasetScheme databaseSchema = new DatasetScheme("data/UCI/Medical/hepatitis.arff", ".arff", folds);
		//DatasetScheme databaseSchema = new DatasetScheme("data/UCI/Medical/hypothyroid.arff", ".arff", folds);
		//DatasetScheme databaseSchema = new DatasetScheme("data/UCI/Medical/liver-disorders.arff", ".arff", folds);
		//DatasetScheme databaseSchema = new DatasetScheme("data/UCI/Medical/lung-cancer.arff", ".arff", folds);
		//DatasetScheme databaseSchema = new DatasetScheme("data/UCI/Medical/lymph.arff", ".arff", folds);
		//DatasetScheme databaseSchema = new DatasetScheme("data/UCI/thyroid/new-thyroid.data", ".data", folds);
		
		
		//IMBALANCED DATASETS
		//DatasetScheme databaseSchema = new DatasetScheme("data/UCI/Medical/ThoraricSurgery.arff", ".arff", folds);
		DatasetScheme databaseSchema = new DatasetScheme("data/UCI/thyroid/sick-euthyroid.data", ".data", folds);
		//DatasetScheme databaseSchema = new DatasetScheme("data/UCI/pageblocks/page-blocks.data", ".data", folds);
		//DatasetScheme databaseSchema = new DatasetScheme("data/UCI/glass/glass.data", ".data", folds);
		//DatasetScheme databaseSchema = new DatasetScheme("data/UCI/yeast/yeast.data", ".data", folds);
		//DatasetScheme databaseSchema = new DatasetScheme("data/UCI/Medical/ecoli.arff", ".arff", folds);
		//DatasetScheme databaseSchema = new DatasetScheme("data/UCI/abalone/abalone.data", ".data", folds);
		

		//System.setOut(new PrintStream(new BufferedOutputStream(new FileOutputStream("out\\"+databaseSchema.toString()+" "+experimentSchema.toString()+".txt")), true));
		
		//CROSS VALIDATION
	    Instances instances = databaseSchema.getAllTestInstances();
	    Evaluation eval = new Evaluation(instances);
	    if(databaseSchema.getFolds()>1)
	    	eval.crossValidateModel(experimentSchema.produceClassifier(), instances, databaseSchema.getFolds(), new java.util.Random(1));
	    else
	    	for (int n = 0; n < databaseSchema.getFolds(); n++) {
				Classifier classifier = experimentSchema.produceClassifier();
				System.out.print("Cross-validation "+n+" ..");
				databaseSchema.produceNextSets(n);
				Instances train = databaseSchema.getTrainSet();
				Instances test = databaseSchema.getTestSet();
				System.out.print(" ("+train.numInstances()+" training, "+test.numInstances()+" test instances) ");
				System.out.print(" building.. ");
				classifier.buildClassifier(train);
				System.out.print(" evaluating.. ");
				eval.evaluateModel(classifier, test);
				System.out.println(" finished");
			}
		
		
		//System.out.println(eval.toMatrixString());
	    System.out.println("\n================ Individual Report");
	    //System.out.println(eval.toSummaryString());
		double[] classFrequencies = eval.getClassPriors();
		double sumFrequencies = 0;
		for(int i=0;i<classFrequencies.length;i++)
			sumFrequencies += classFrequencies[i];
		for(int i=0;i<classFrequencies.length;i++)
			classFrequencies[i] /= sumFrequencies;
		double meanTPr = 0;
		double GTPr = 1;
	    for(int i=0;i<instances.numClasses();i++)  {
		    System.out.println(toLength("TPR "+instances.classAttribute().value(i), 35)+toPercentage(eval.truePositiveRate(i))+"\t   ("+toPercentage(classFrequencies[i])+" presence)");
		    meanTPr += eval.truePositiveRate(i)/instances.numClasses();
		    GTPr *= Math.pow(eval.truePositiveRate(i),1.0/instances.numClasses());
	    }
	    System.out.println(toLength("AM for "+instances.classAttribute(), 35)+toPercentage(meanTPr));
	    System.out.println(toLength("GM for "+instances.classAttribute(), 35)+toPercentage(GTPr));
	    
		System.out.println("\n================ Global Report");
	    System.out.println(toLength("Weighted AUC", 25)+toPercentage(eval.weightedAreaUnderROC(), 0));
	    System.out.println(toLength("Weighted TPR", 25)+toPercentage(eval.weightedTruePositiveRate(), 0));
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
	    System.out.println(toLength("Fairness", 25)+toPercentage(fairNom/fairDenom, 0));
	    System.out.println(toLength("Fairness * Weighted TPr", 25)+toPercentage(fairNom/fairDenom*eval.weightedTruePositiveRate(), 0));
	    
		java.awt.Toolkit.getDefaultToolkit().beep();
	}
	private static String toLength(String str, int len) {
		String ret = str;
		while(ret.length()<len)
			ret += " ";
		return ret;
	}
	private static String toPercentage(double val, int spaces) {
		if(spaces==0)
			return ""+(int)Math.round(val*100)+"%";
		else
			return ""+Math.round(val*Math.pow(10, spaces+2))/Math.pow(10, spaces)+"%";
	}
	private static String toPercentage(double val) {
		return ""+Math.round(val*100)+"%";
	}

}
