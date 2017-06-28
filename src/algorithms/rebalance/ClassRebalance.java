package algorithms.rebalance;
import java.io.OutputStream;
import java.io.PrintStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;

/**
 * <h1>ClassRebalance</h1>
 * This weka {@link Classifier} performs rebalancing that increases fairness of a base classifier for 
 * imbalanced problem classification. The base classifier is trained alongside this class (no separate
 * training needed).<br/>
 * Options can either be parsed through the constructor or through {@link #setOptions(String[])};<br/>
 * <h2>-fun[ction]</h2>
 * Selects the appropriate rebalance function (default value is <cose>exponential</code>);<br/>
 * <code>none</code> : does not perform rebalancing <br/>
 * <code>lin[ear]</code> : uses the rebalance function(1-f)*w <br/>
 * <code>exp[onential]</code> : uses the rebalance function w^f <br/>
 * <code>inv[erse]</code> : uses the rebalance function (w*min(f)/f)^(1/3.5) <br/>
 * <h2>-pre[process]</h2>
 * <code>[re]sample</code> : performs resamling before training <br/>
 * <code>SMOTE</code> : performs SMOTE before training
 * <h2>-rebalance</h1>
 * Sets the rebalance parameter <code>a</code>, performing rebalance as <code>w = (1-a)w + a function(w,f)</code>.<br/>
 * For resampling, this values determines the bias towards a normal distribution.
 * Default value is <code>0.8</code>.
 * <h2>-dynamic</h2>
 * Selects dynamic rebalance selection options (default value is <code>entropy</code>;<br/>
 * <code>none</code> : the rebalance parameter is a constant <br/>
 * <code>entropy</code> : before performing rebalancing at each step, the rebalance parameter is multiplied
 * with normalized weight entropy across weights for selection, so that stronger classifications (i.e.
 * more extreme values and thus lower weight entropies) would not get rebalanced as much. <br/>
 * <code>margin</code> : substract the rebalance function from its max for the particular class frequency
 * <code>clean[liness]</code> : multiply rebalance parameter with 1-cleanliness of the sample's neighborhood
 * <h2>-pos[itive] | -neg[ative]</h2>
 * Sets rebalance to either positive or negative one (is positive by default).<br/>
 * For undersampling or oversampling, always use positive rebalance. For other case, rebalance parameter
 * sign can also determine positive or negative rebalance if not otherwise specified.
 * <h2>-trained</h2>
 * Denotes that the base classifier does not need training when calling {@link #buildClassifier}. Should be
 * used <b>only</b> when trying to avoid retraining when calculating parameters for weight rebalance.
 * <h2>-sensitive</h2>
 * Uses the log function to make original classifications more sensitive and uncertain.
 * <h2>-debug</h2>
 * Send some detailed information to <code>System.out</code>.
 * 
 * @author Krasanakis Emmanouil
 */
public class ClassRebalance extends Classifier implements Serializable {
	private static final long serialVersionUID = -7184425438333439884L;
	private static enum FunctionForm {none, linear, exponential, inverse, log, thresholding};
	private static enum PreprocessForm {none, resample, SMOTE};
	private static enum DynamicForm {none, entropy, margin, max, cleanliness};
	private double[] frequencies;
	private double minFrequency = 1;
	private double maxFrequency = 1;
	private double rebalanceParameter = 0;
	private DynamicForm dynamicForm = DynamicForm.none;
	private FunctionForm functionForm = FunctionForm.none;
	private PreprocessForm preprocessForm = PreprocessForm.none;
	private Classifier baseClassifier;
	private boolean pretrained = false;
	private boolean debug = false;
	private double tune = 0;
	private int deepDebug = 1;//level 1 debugs tuning, level 2 also debugs instances, level 0 does not print those things
	private double sensitive = 0;
	private int tuneFolds = 3;
	public static boolean throtleBaseClassifierPrints = true;
	
	public ClassRebalance(Classifier baseClassifier, String[] options) throws Exception  {
		this.baseClassifier = baseClassifier;
		setOptions(options);
	}
	public ClassRebalance(Classifier baseClassifier) throws Exception {
		this(baseClassifier, weka.core.Utils.splitOptions("-function exp -rebalance 0.8 -positive -dynamic entropy"));
	}
	@Override
	public void setOptions(String[] options) throws Exception {
		super.setOptions(options);
		for(int i=0;i<options.length;i++)
			options[i] = options[i].toLowerCase();
		int rebalanceSign = 0;
		for(int i=0;i<options.length;i++) {
			if(options[i].startsWith("-trained")) {
				pretrained = true;
			}
			else if(options[i].startsWith("-debug")) {
				debug = true;
			}
			else if(options[i].startsWith("-sensitive")) {
				i++;
				sensitive = Double.parseDouble(options[i]);
			}
			else if(options[i].startsWith("-fun")) {
				i++;
				if(options[i].startsWith("exp"))
					functionForm = FunctionForm.exponential;
				else if(options[i].startsWith("thr"))
					functionForm = FunctionForm.thresholding;
				else if(options[i].startsWith("inv"))
					functionForm = FunctionForm.inverse;
				else if(options[i].startsWith("lin"))
					functionForm = FunctionForm.linear;
				else if(options[i].startsWith("log"))
					functionForm = FunctionForm.log;
				else if(options[i].startsWith("non"))
					functionForm = FunctionForm.none;
				else
					throw new Exception("Invalid -function option");
			}
			else if(options[i].startsWith("-pre")) {
				i++;
				if(options[i].endsWith("sample"))
					preprocessForm = PreprocessForm.resample;
				else if(options[i].startsWith("smote"))
					preprocessForm = PreprocessForm.SMOTE;
				else if(options[i].startsWith("non"))
					preprocessForm = PreprocessForm.none;
				else
					throw new Exception("Invalid -preprocess option");
			}
			else if(options[i].startsWith("-reb")) {
				i++;
				tune = 0;
				if(options[i].startsWith("tune"))
					tune = 1;
				else
					rebalanceParameter = Double.parseDouble(options[i]);
			}
			else if(options[i].startsWith("-pos")) {
				rebalanceSign = 1;
			}
			else if(options[i].startsWith("-neg")) {
				rebalanceSign = -1;
			}
			else if(options[i].startsWith("-dyn")) {
				i++;
				if(options[i].startsWith("entr"))
					dynamicForm = DynamicForm.entropy;
				else if(options[i].startsWith("non"))
					dynamicForm = DynamicForm.none;
				else if(options[i].startsWith("margin"))
					dynamicForm = DynamicForm.margin;
				else if(options[i].startsWith("max"))
					dynamicForm = DynamicForm.max;
				else if(options[i].startsWith("clean"))
					dynamicForm = DynamicForm.cleanliness;
				else
					throw new Exception("Invalid -dynamic option");
			}
			else
				throw new Exception("Unknown or malformed rebalance options");
		}
		if(rebalanceSign!=0)
			rebalanceParameter = rebalanceSign*Math.abs(rebalanceParameter);
	}
	public static double getImbalance(double[] frequencies) {
		double imbalance = 0;
		for(int i=0;i<frequencies.length;i++)
			for(int j=0;j<frequencies.length;j++)
				if(i!=j)
					imbalance += frequencies[i]/frequencies[j];
		imbalance /= frequencies.length/(frequencies.length-1);
		return 2*imbalance;
	}
	private Instances trainingInstances;
	@Override
	public void buildClassifier(Instances instances) throws Exception {
		if(debug) {
			System.out.println("\nRebalance method        : "+functionForm.toString());
			System.out.println("Preprocessing           : "+preprocessForm.toString());
			System.out.println("Dynamic                 : "+dynamicForm.toString());
		}
		frequencies = DatasetMetrics.getPriors(instances);
		minFrequency = 1;
		maxFrequency = 0;
		for(double freq : frequencies)
			if(freq<minFrequency)
				minFrequency = freq;
		for(double freq : frequencies)
			if(freq>maxFrequency)
				maxFrequency = freq;
		if(debug) {
			System.out.println("Frequencies             : "+java.util.Arrays.toString(frequencies));
			System.out.println("Imbalance               : "+getImbalance(frequencies));
		}
		Instances trainingInstances = instances;
		if(preprocessForm==PreprocessForm.resample) {
			weka.filters.supervised.instance.Resample res = new weka.filters.supervised.instance.Resample();
			res.setInputFormat(instances);
			if(maxFrequency/minFrequency<10)
				res.setSampleSizePercent(50);
			else
				res.setSampleSizePercent(100);
			res.setBiasToUniformClass(1);
			trainingInstances = Filter.useFilter(trainingInstances, res);
			double[] newFrequencies = DatasetMetrics.getPriors(trainingInstances);
			if(debug) {
				System.out.println("New Frequencies         : "+java.util.Arrays.toString(newFrequencies));
				System.out.println("New Imbalance           : "+getImbalance(newFrequencies));
			}
		}
		if(preprocessForm==PreprocessForm.SMOTE) {
			weka.filters.supervised.instance.SMOTE res = new weka.filters.supervised.instance.SMOTE();
			res.setInputFormat(trainingInstances);
			trainingInstances = Filter.useFilter(trainingInstances, res);
		}
		this.trainingInstances = trainingInstances;
		//build base classifier
		if(!pretrained) {
			if(debug)
				System.out.println("Training base classifier: "+baseClassifier.getClass().getSimpleName());
			if(throtleBaseClassifierPrints && !debug) {
			    PrintStream original = System.out;
			    System.setOut(new PrintStream(new OutputStream() {
	                public void write(int b) {
	                    //DO NOTHING
	                }
	            }));
				baseClassifier.buildClassifier(trainingInstances);
			    System.setOut(original);
			}
			else
				baseClassifier.buildClassifier(trainingInstances);
		}
		else if(debug)
			System.out.println("Pretrained base classifier: "+baseClassifier.getClass().getSimpleName());
		if(debug) {
			if(tune==0)
				System.out.println("Rebalance parameter     : "+rebalanceParameter);
			else
				System.out.print("Rebalance parameter     : TUNE ");
		}
		//tune rebalance parameter
		if(tune!=0) {
			performTuning(instances, -2, 2, 2, tuneFolds);
			if(debug)
				System.out.println();
		}
	}
	protected double performTuning(Instances trainingInstances, double minRebalanceParameter, double maxRebalanceParameter, int depth, int folds) throws Exception {
		double bestParameter = 0;
		double bestParameterFairness = 0;
		double prevTune = tune;
		boolean prevDebug = debug;
		tune = 0;
		debug = false;
		Classifier previousClassifier = baseClassifier;
		double step = (maxRebalanceParameter-minRebalanceParameter)/10;
		for(double val=minRebalanceParameter;val<=maxRebalanceParameter;val+=step) {
			rebalanceParameter = val;
			double performance = 1;
			double fairness = 0;
			double fairnessDenom = 0;
			if(folds==1) {
				Evaluation eval = new Evaluation(trainingInstances);
				eval.evaluateModel(this,trainingInstances);
				double[] priors = eval.getClassPriors();
				//System.out.println("\nPerrformance for "+rebalanceParameter);
				for(int i=0;i<priors.length;i++) {
					performance *= Math.pow(eval.truePositiveRate(i), 1.0/priors.length);
					//System.out.println("class "+trainingInstances.classAttribute().value(i)+" ("+eval.getClassPriors()[i]+") : "+eval.truePositiveRate(i));
					for(int j=0;j<priors.length;j++) 
						if(i!=j){
							fairness += Math.abs(eval.truePositiveRate(i)-eval.truePositiveRate(j));
							fairnessDenom += 1;
						}
				}
				fairness = 0.2*(1-fairness/fairnessDenom)+performance;
				//System.out.println("Overall: "+fairness);
			}
			else if(!pretrained) {
				Evaluation eval = new Evaluation(trainingInstances);
				eval.crossValidateModel(makeCopy(this), trainingInstances, folds, new Random(1));
				double[] priors = eval.getClassPriors();
				for(int i=0;i<priors.length;i++) {
					performance *= Math.pow(eval.truePositiveRate(i), 1.0/priors.length);
					for(int j=0;j<priors.length;j++) 
						if(i!=j){
							fairness += Math.abs(eval.truePositiveRate(i)-eval.truePositiveRate(j));
							fairnessDenom += 1;
						}
				}
				fairness = 0.2*(1-fairness/fairnessDenom)+performance;
			}
			else
				throw new RuntimeException("Cannot tune rebalance when using pretrained classifiers");
			if(fairness>bestParameterFairness) {
				bestParameterFairness = fairness;
				bestParameter = val;
			}
		}
		debug = prevDebug;
		tune = prevTune;
		baseClassifier = previousClassifier;
		rebalanceParameter = bestParameter;
		
		
		if(debug && (deepDebug>=1 || depth==0))
			System.out.print(rebalanceParameter+" ("+bestParameterFairness+") ");
		//if(debug && deepDebug>=1)
			//System.out.print("\nIncreased fairness to   : "+bestParameterFairness+" for parameter: ");

		if(depth>0) {
			double evalCurrent = performTuning(trainingInstances, rebalanceParameter-step, rebalanceParameter+step, depth-1, folds);
		}
		
		return bestParameterFairness;
	}
	protected double obtainTrainingScore(Instances instances) throws Exception {
		double[] TP = new double[instances.numClasses()];
		double[] P = new double[instances.numClasses()];
		for(int i=0;i<instances.numInstances();i++) {
			Instance instance = instances.instance(i);
			int classifiedInto = (int)classifyInstance(instance);
			if(classifiedInto==-1)
				continue;
			P[classifiedInto]+=1;
			if(classifiedInto==(int)instance.classValue())
				TP[classifiedInto]+=1;
		}
		//TP now contains TPr
		for(int i=0;i<TP.length;i++)
			if(P[i]!=0)
				TP[i] /= P[i];
		//return TPr fairness
		double res = 1;
	    for(int i=0;i<TP.length;i++) 
	    	res *= TP[i];
	    return Math.pow(res, 1.0/TP.length);
	}
	protected double rebalanceDerivativeT(double f, double w, double positive) {
		if(functionForm==FunctionForm.exponential)
			return ((1-positive)/2+positive*f)*Math.pow(w, (1-positive)/2+positive*f-1);
		else if(functionForm==FunctionForm.linear)
			return ((1+positive)/2-positive*f);
		else if(functionForm==FunctionForm.inverse) 
			return Math.pow(f==0?1:f, -positive);// /3.5
		else
			throw new RuntimeException("Rebalance function derivative not implemented");
		//return f;
	}
	protected double rebalanceFunctionT(double f, double w, double positive) {
		if(functionForm==FunctionForm.exponential)
			return Math.pow(w, (1-positive)/2+positive*f);
		else if(functionForm==FunctionForm.linear)
			return Math.pow(w, 1) * ((1+positive)/2-positive*f);
		else if(functionForm==FunctionForm.log)
			return Math.log(1+w)*((1+positive)/2-positive*f);
		else if(functionForm==FunctionForm.inverse) 
			return w * Math.pow(f==0?1:f, -positive);// /3.5
		else if(functionForm==FunctionForm.thresholding)
			return ((1+positive)-positive*f);
		return w;
	}
	protected double rebalanceFunction(double f, double w, double positive, double rebalanceConstant) {
		if(dynamicForm==DynamicForm.max) 
			rebalanceConstant = rebalanceConstant/rebalanceDerivativeT(maxFrequency, 1, 1);
		//else
			//rebalanceConstant = rebalanceConstant/rebalanceFunctionT(minFrequency, 1, 1);
		return w*(1-rebalanceConstant) + rebalanceConstant*rebalanceFunctionT(f, w, positive);
	}
	protected ArrayList<Instance> getNearestInstances(Instances instances, int k, Instance center) {
		ArrayList<Instance> candidates = new ArrayList<Instance>();
		ArrayList<Instance> found = new ArrayList<Instance>();
		for(int i=0;i<instances.numInstances();i++) 
			candidates.add(instances.instance(i));
		while(k>0) {
			double minDistance = Double.POSITIVE_INFINITY;
			Instance minInstance = null;
			for(Instance instance : candidates) {
				double distance = 0;
				for(int i=0;i<instance.numAttributes();i++)
					if(i!=instance.classIndex())
						distance += Math.abs(instance.value(i)-center.value(i));
				//distance /= instances.numAttributes()-1;
				if(distance<minDistance) {
					minDistance = distance;
					minInstance = instance;
				}
			}
			candidates.remove(minInstance);
			found.add(minInstance);
			k--;
		}
		return found;
	}
	protected double getNeighborhoodCleanliness(Instance center, int k) {
		ArrayList<Instance> candidates = getNearestInstances(trainingInstances, k, center);
		double val = 0;
		for(Instance instance1 : candidates)
			for(Instance instance2 : candidates)
				val += instance1.classValue()==instance2.classValue()?1:0;
		val /= k*k;
		return val;
	}
	@Override
	public double[] distributionForInstance(Instance instance) throws Exception {
		//get original distribution
		double[] distribution = baseClassifier.distributionForInstance(instance);
		if(deepDebug>=2)
			System.out.println("Original: "+java.util.Arrays.toString(distribution));
		if(rebalanceParameter==0)
			return distribution;
		//extract rebalance parameter
		double entropyEnhancement = Math.abs(rebalanceParameter);
		double positive = rebalanceParameter<0?-1:1;
		//calculate normalized entropy
		if(sensitive!=0 && normalizedEntropy(distribution)<sensitive) {
			logitize(distribution);
			if(deepDebug>=2)
				System.out.println("Sensitive: "+java.util.Arrays.toString(distribution));
		}
		double normalizedEntropy = 1;
		if(dynamicForm==DynamicForm.entropy || dynamicForm==DynamicForm.margin) 
			normalizedEntropy = normalizedEntropy(distribution);
		if(dynamicForm==DynamicForm.cleanliness) 
			normalizedEntropy *= 1-getNeighborhoodCleanliness(instance, 5);
		normalizedEntropy *= entropyEnhancement;
		//perform rebalancing
		if(dynamicForm==DynamicForm.margin) 
			for(int i=0;i<distribution.length;i++) 
				distribution[i] = rebalanceFunction(frequencies[i], distribution[i], positive, normalizedEntropy)-rebalanceFunction(frequencies[i], 1, positive, normalizedEntropy)+1;
		else
			for(int i=0;i<distribution.length;i++) 
				distribution[i] = Math.max(rebalanceFunction(frequencies[i], distribution[i], positive, normalizedEntropy), 0);
		//if(functionForm==FunctionForm.inverse) {
			double sum = 0;
			for(int i=0;i<distribution.length;i++) 
				sum += distribution[i];
			if(sum!=0)
			for(int i=0;i<distribution.length;i++) 
				distribution[i] /= sum;
		//}
		if(deepDebug>=2)
			System.out.println("Changed to: "+java.util.Arrays.toString(distribution)+"\n");
		return distribution;
	}
	@Override
	public double classifyInstance(Instance instance) throws Exception {
		double maxClassification = Double.NEGATIVE_INFINITY;
		int selection = -1;
		double[] distribution = distributionForInstance(instance);
		for(int i=0;i<distribution.length;i++) 
			if(maxClassification<=distribution[i]) {
				maxClassification = distribution[i];
				selection = i;
			}
		return selection;
	}
	public static double normalizedEntropy(double[] distribution) {
		double normalizedEntropy = 0;
		for(int i=0;i<distribution.length;i++)
			if(distribution[i]!=0)
				normalizedEntropy  -= Math.log(distribution[i])*distribution[i];
		return normalizedEntropy/Math.log(distribution.length);
	}
	protected void logitize(double[] distribution) {
		boolean hasOne = false;
		for(int i=0;i<distribution.length;i++)
			if(distribution[i]==1)
				hasOne = true;
		if(!hasOne) {
			double epsilon = 10.0/frequencies.length;
			double min = Double.POSITIVE_INFINITY;
			for(int i=0;i<distribution.length;i++)
				if(distribution[i]!=0)
					min = Math.min(min, distribution[i] = Math.log(distribution[i]/(1-distribution[i])));
			for(int i=0;i<distribution.length;i++)
				if(distribution[i]!=0)
					distribution[i] -= min-epsilon;
			double sum = 0;
			for(int i=0;i<distribution.length;i++)
				sum += distribution[i];
			if(sum!=0)
				for(int i=0;i<distribution.length;i++)
					distribution[i] /= sum;
		}
	}
}
