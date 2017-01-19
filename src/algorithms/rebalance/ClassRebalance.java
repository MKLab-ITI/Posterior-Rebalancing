package algorithms.rebalance;
import java.io.OutputStream;
import java.io.PrintStream;
import java.io.Serializable;

import weka.classifiers.Classifier;
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
	private static enum FunctionForm {none, linear, exponential, inverse};
	private static enum PreprocessForm {none, resample, SMOTE};
	private static enum DynamicForm {none, entropy, margin};
	private double[] frequencies;
	private double rebalanceParameter = 0;
	private DynamicForm dynamicForm = DynamicForm.none;
	private FunctionForm functionForm = FunctionForm.none;
	private PreprocessForm preprocessForm = PreprocessForm.none;
	private Classifier baseClassifier;
	private boolean pretrained = false;
	private boolean debug = false;
	private boolean tune = false;
	private int deepDebug = 1;//level 1 debugs tuning, level 2 also debugs instances, level 0 does not print those things
	private double sensitive = 0;
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
				else if(options[i].startsWith("inv"))
					functionForm = FunctionForm.inverse;
				else if(options[i].startsWith("lin"))
					functionForm = FunctionForm.linear;
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
				tune = options[i].startsWith("tune");
				if(!tune)
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
				else
					throw new Exception("Invalid -dynamic option");
			}
			else
				throw new Exception("Unknown or malformed rebalance options");
		}
		if(rebalanceSign!=0)
			rebalanceParameter = rebalanceSign*Math.abs(rebalanceParameter);
	}
	public static double[] getPriors(Instances instances) {
		//obtain frequencies
		double[] frequencies = new double[instances.numClasses()];
		for(int i=0;i<instances.numInstances();i++) {
			Instance instance = instances.instance(i);
			frequencies[(int)instance.classValue()]++;
		}
		//convert to probabilities
		double sum = 0;
		for(int i=0;i<frequencies.length;i++)
			sum += frequencies[i];
		if(sum!=0)
			for(int i=0;i<frequencies.length;i++)
				frequencies[i] /= sum;
		return frequencies;
	}
	public static double getImbalance(double[] frequencies) {
		double imbalance = 0;
		for(int i=0;i<frequencies.length;i++)
			for(int j=0;j<frequencies.length;j++)
				if(i!=j)
					imbalance += frequencies[i]/frequencies[j];
		imbalance /= frequencies.length/(frequencies.length-1);
		return imbalance;
	}
	@Override
	public void buildClassifier(Instances instances) throws Exception {
		if(debug) {
			System.out.println("\nRebalance method        : "+functionForm.toString());
			System.out.println("Preprocessing           : "+preprocessForm.toString());
			System.out.println("Dynamic                 : "+dynamicForm.toString());
		}
		frequencies = getPriors(instances);
		if(debug) {
			System.out.println("Frequencies             : "+java.util.Arrays.toString(frequencies));
			System.out.println("Imbalance               : "+getImbalance(frequencies));
		}
		Instances trainingInstances = instances;
		if(preprocessForm==PreprocessForm.resample) {
			weka.filters.supervised.instance.Resample res = new weka.filters.supervised.instance.Resample();
			res.setInputFormat(instances);
			res.setSampleSizePercent(100);
			res.setBiasToUniformClass(1);
			trainingInstances = Filter.useFilter(trainingInstances, res);
			double[] newFrequencies = getPriors(trainingInstances);
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
		if(debug) {
			if(!tune)
				System.out.println("Rebalance parameter     : "+rebalanceParameter);
			else
				System.out.print("Rebalance parameter     : TUNE ");
		}
		//tune rebalance parameter
		if(tune)
			performTuning(instances, -1, 1);
	}
	protected void performTuning(Instances trainingInstances, double minRebalanceParameter, double maxRebalanceParameter) throws Exception {
		rebalanceParameter = (minRebalanceParameter+maxRebalanceParameter)/2;
		double r = (-minRebalanceParameter+maxRebalanceParameter)/4;
		int direction = 0;
		double prevFairness = obtainFairness(trainingInstances, false);
		while(true) {				
			double r_sup = Math.min(Math.abs(minRebalanceParameter-rebalanceParameter), Math.abs(maxRebalanceParameter-rebalanceParameter));
			double r_left = Math.min(r, r_sup)*(direction==-1?2:1)/2;
			double r_right = Math.min(r, r_sup)*(direction==1?2:1)/2;
			double nextLeft = rebalanceParameter-r_left;
			double nextRight = rebalanceParameter+r_right;
			double bias = (rebalanceParameter<0?-0.001:0.001)*r;
			rebalanceParameter = nextLeft;
			double leftFairness = obtainFairness(trainingInstances, false);
			rebalanceParameter = nextRight;
			double rightFairness = obtainFairness(trainingInstances, false);
			if(leftFairness<rightFairness+bias && rightFairness+bias>=prevFairness) {
				rebalanceParameter = nextRight;
				r = r_right;
				direction = 1;
				prevFairness = rightFairness;
			}
			else if(leftFairness>rightFairness+bias && leftFairness+bias>=prevFairness) {
				rebalanceParameter = nextLeft;
				r = r_left;
				direction = -1;
				prevFairness = leftFairness;
			}
			else //if(direction==0)
				r /= 2;
			//else 
				//direction = 0;
			if(/*Math.abs(leftFairness-rightFairness-rightFairnessOffset)<0.005 ||*/r<0.02 || rebalanceParameter>=maxRebalanceParameter || rebalanceParameter<=minRebalanceParameter)
				break;
			if(debug && deepDebug>=1)
				System.out.print(rebalanceParameter+" ("+prevFairness+") ");
		}
		if(debug && deepDebug>=1)
			System.out.print(rebalanceParameter+" ("+prevFairness+") ");
		if(debug && deepDebug>=1)
			System.out.print("\nIncreased fairness to   : "+prevFairness+" for parameter: ");
		if(debug)
			System.out.println(+rebalanceParameter);
	}
	protected double obtainFairness(Instances instances, boolean signed) throws Exception {
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
		double fairNom = 0;
		double fairDenom = 0;
	    for(int i=0;i<TP.length;i++) 
		    for(int j=0;j<TP.length;j++) {
		    	double weight = frequencies[i]*frequencies[j];
		    	if(i!=j)
		    		fairDenom += weight;
		    	//if(signed && (TP[i]-TP[j])*(frequencies[i]-frequencies[j])<0)
		    		//weight = -weight;
		    	if(i!=j && TP[i]!=0 && TP[j]!=0) {
		    		double coupleUnfairness = TP[i]/TP[j]+TP[j]/TP[i];
		    		fairNom += 2*weight/coupleUnfairness;
		    	}
		    }
	    return fairNom/fairDenom;
	}
	protected double rebalanceFunction(double f, double w, double positive, double rebalanceConstant) {
		if(functionForm==FunctionForm.exponential)
			return w*(1-rebalanceConstant)+rebalanceConstant*Math.pow(w, (1-positive)/2+positive*f);
		else if(functionForm==FunctionForm.linear)
			return w*(1-rebalanceConstant)+rebalanceConstant*w * ((1+positive)/2-positive*f);
		else if(functionForm==FunctionForm.inverse) 
			return w*(1-rebalanceConstant)+rebalanceConstant * w * Math.pow(f==0?1:f, -positive);// /3.5
		return w;
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
		if(dynamicForm!=DynamicForm.none) 
			normalizedEntropy = normalizedEntropy(distribution);
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
