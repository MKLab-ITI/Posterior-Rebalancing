import java.awt.BorderLayout;
import java.io.BufferedOutputStream;
import java.io.FileOutputStream;
import java.io.PrintStream;

import algorithms.rebalance.ExperimentScheme;
import importer.DatasetScheme;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.ThresholdCurve;
import weka.core.Instances;
import weka.core.Utils;
import weka.gui.visualize.PlotData2D;
import weka.gui.visualize.ThresholdVisualizePanel;

public class Main {
	public static void main(String[] args) throws Exception {
		//IMPORT INSTANCES
		ExperimentScheme experimentSchema = new ExperimentScheme("Logistic", "", "DTthr");
		int folds = 10;
		
		//OTHER DATASETS
		//DatasetScheme databaseSchema = new DatasetScheme("sexuality", "myPersonality", folds);
		//DatasetScheme databaseSchema = new DatasetScheme("religion", "myPersonality", folds);
		//DatasetScheme databaseSchema = new DatasetScheme("politics", "myPersonality", folds);
		//DatasetScheme databaseSchema = new DatasetScheme("sex", "usemp250", folds);//don't forget to also make -sensitive 1 or 2 for usemp250
		//DatasetScheme databaseSchema = new DatasetScheme("relig", "usemp250", folds);
		//DatasetScheme databaseSchema = new DatasetScheme("polit", "usemp250", folds);
		//DatasetScheme databaseSchema = new DatasetScheme("cannabis", "usemp250", folds);
		//DatasetScheme databaseSchema = new DatasetScheme("spam", "UCI", folds);
		//DatasetScheme databaseSchema = new DatasetScheme("cell", "UCI", folds);
		//DatasetScheme databaseSchema = new DatasetScheme("data/UCI/Medical/diabetes.arff", ".arff", folds);
		//DatasetScheme databaseSchema = new DatasetScheme("data/UCI/Medical/hepatitis.arff", ".arff", folds);
		//DatasetScheme databaseSchema = new DatasetScheme("data/UCI/Medical/hypothyroid.arff", ".arff", folds);
		//DatasetScheme databaseSchema = new DatasetScheme("data/UCI/Medical/liver-disorders.arff", ".arff", folds);
		//DatasetScheme databaseSchema = new DatasetScheme("data/UCI/Medical/lymph.arff", ".arff", folds);
		//DatasetScheme databaseSchema = new DatasetScheme("data/UCI/thyroid/new-thyroid.data", ".data", folds);
		//DatasetScheme databaseSchema = new DatasetScheme("data/UCI/abalone.data", ".data", folds);
		//DatasetScheme databaseSchema = new DatasetScheme("data/UCI/yeast.data", ".data", folds);
		//DatasetScheme databaseSchema = new DatasetScheme("data/UCI/wine.data", ".data", folds);
		
		
		//IMBALANCED DATASETS
		//DatasetScheme databaseSchema = new DatasetScheme("data/UCI/adult.data", ".data", folds);
		DatasetScheme databaseSchema = new DatasetScheme("data/UCI/car.data", ".data", folds);
		//DatasetScheme databaseSchema = new DatasetScheme("data/UCI/cmc.data", ".data", folds);
		//DatasetScheme databaseSchema = new DatasetScheme("data/UCI/glass.data", ".data", folds);
		//DatasetScheme databaseSchema = new DatasetScheme("data/UCI/Medical/lung-cancer.arff", ".arff", folds);
		//DatasetScheme databaseSchema = new DatasetScheme("data/UCI/page-blocks.data", ".data", folds);
		//DatasetScheme databaseSchema = new DatasetScheme("data/UCI/sick-euthyroid.data", ".data", folds);
		//DatasetScheme databaseSchema = new DatasetScheme("data/UCI/ThoraricSurgery.arff", ".arff", folds);
		//DatasetScheme databaseSchema = new DatasetScheme("data/UCI/yeast.data", ".data", folds);//use D1lin
		
		
		
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
		
		
		//System.out.println(eval.toClassDetailsString());
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
		int count = 0;
	    for(int i=0;i<instances.numClasses();i++)  {
	    	if(classFrequencies[i]==0)
	    		continue;
		    System.out.println(toLength("TPR "+instances.classAttribute().value(i), 35)+toPercentage(eval.truePositiveRate(i))+"\t   ("+toPercentage(classFrequencies[i])+" presence)");
		    meanTPr += eval.truePositiveRate(i)/instances.numClasses();
		    count++;
		    GTPr *= eval.truePositiveRate(i);
	    }
	    GTPr =  Math.pow(GTPr,1.0/count);
	    //System.out.println(toLength("AM for "+instances.classAttribute(), 35)+toPercentage(meanTPr));
	    System.out.println(toLength("GM for "+instances.classAttribute(), 35)+toPercentage(GTPr));
	    
		//System.out.println("\n================ Global Report");
	    System.out.println(toLength("Weighted AUC", 25)+toPercentage(eval.weightedAreaUnderROC(), 0));
	    //System.out.println(toLength("Weighted TPR", 25)+toPercentage(eval.weightedTruePositiveRate(), 0));
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
	    System.out.println(toLength("Imbalance", 25)+toPercentage(fairNom/fairDenom, 1));
	    //System.out.println(toLength("Fairness * Weighted TPr", 25)+toPercentage(1-fairNom/fairDenom*eval.weightedTruePositiveRate(), 0));
	    
	    
	    
	    
	    
	    
	    // generate curve
	    /* ThresholdCurve tc = new ThresholdCurve();
	     int classIndex = 0;
	     Instances result = tc.getCurve(eval.predictions(), classIndex);
	 
	     // plot curve
	     ThresholdVisualizePanel vmc = new ThresholdVisualizePanel();
	     vmc.setROCString("(Area under ROC = " +
	         Utils.doubleToString(tc.getROCArea(result), 4) + ")");
	     vmc.setName(result.relationName());
	     PlotData2D tempd = new PlotData2D(result);
	     tempd.setPlotName(result.relationName());
	     tempd.addInstanceNumberAttribute();
	     // specify which points are connected
	     boolean[] cp = new boolean[result.numInstances()];
	     for (int n = 1; n < cp.length; n++)
	       cp[n] = true;
	     tempd.setConnectPoints(cp);
	     // add plot
	     vmc.addPlot(tempd);
	 
	     // display curve
	     String plotName = vmc.getName();
	     final javax.swing.JFrame jf =
	       new javax.swing.JFrame("Weka Classifier Visualize: "+plotName);
	     jf.setSize(500,400);
	     jf.getContentPane().setLayout(new BorderLayout());
	     jf.getContentPane().add(vmc, BorderLayout.CENTER);
	     jf.addWindowListener(new java.awt.event.WindowAdapter() {
	       public void windowClosing(java.awt.event.WindowEvent e) {
	       jf.dispose();
	       }
	     });
	     jf.setVisible(true);
	    
	    
	    */
	    
	    
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
