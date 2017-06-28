package algorithms.rebalance;
import algorithms.implementations.ImprovedSmoothKNN;
import weka.classifiers.Classifier;
import weka.classifiers.functions.LibSVM;

/**
 * <h1>ExperimentScheme</h1>
 * This class is used to produce classifiers according to required scheme options. In particular, it is used
 * to wrap a rebalance scheme of {@link algorithms.rebalance.ClassRebalance} arround a base classifier.
 * 
 * @see #produceClassifier()
 * @author Krasanakis Emmanouil
 */
public class ExperimentScheme {
	private String baseClassifierType, baseClassifierOptions;
	private String rebalanceScheme;
	private String name;
	
	/**
	 * The {@link ExperimenScheme} constructor, which requires all scheme parameters.
	 * @param classifierType the base classifier type
	 * @param options options to be sent to the base classifier
	 * @param rebalanceScheme an abbreviation of the rebalance scheme to be used by {@link #convertScheme}
	 */
	public ExperimentScheme(String classifierType, String baseClassifierOptions, String rebalanceScheme) {
		this.baseClassifierType = classifierType;
		this.baseClassifierOptions = baseClassifierOptions;
		this.rebalanceScheme = convertScheme(rebalanceScheme);
		name = classifierType+" "+rebalanceScheme;
	}
	
	public void appendRebalanceOptions(String text) {
		rebalanceScheme += " " + text;
	}
	
	/**
	 * <h1>produceClassifier</h1>
	 * @return a new {@link weka.classifiers.Classifier} instance of the scheme's classifier
	 * @throws Exception if scheme options were invalid
	 */
	public Classifier produceClassifier() throws Exception {
		Classifier classifier;
		if(baseClassifierType.contains("SKNN")){
			classifier = new ImprovedSmoothKNN();
			//classifier = new SmoothKNN(weka.core.Utils.splitOptions("-neighbors 5 -similarity dot -powSim 1 -powWeight 0"));
		}
		else if(baseClassifierType.contains("RUSBoost")){
			classifier = new algorithms.implementations.Boosting.RUSBoost(new weka.classifiers.functions.Logistic(),28);
		}
		else if(baseClassifierType.contains("ClusterBoost")){
			classifier = new algorithms.implementations.Boosting.ClusterBoost(new weka.classifiers.functions.Logistic());
		}
		else if(baseClassifierType.contains("KNN")){
			classifier = new weka.classifiers.lazy.IBk(5);
			classifier.setOptions(weka.core.Utils.splitOptions("-I"));
		}
		else if(baseClassifierType.contains("SVM")) {
			classifier = new algorithms.implementations.JLibSVMWrapper();
			
		}
		else if(baseClassifierType.contains("SMO"))
			classifier = new weka.classifiers.functions.SMO();
		else if(baseClassifierType.contains("Logistic"))
			classifier = new weka.classifiers.functions.Logistic();
		else if(baseClassifierType.contains("Tree"))
			classifier = new weka.classifiers.trees.J48();
		else if(baseClassifierType.contains("Forest")) {
			classifier = new weka.classifiers.trees.RandomForest();
			((weka.classifiers.trees.RandomForest)classifier).setNumTrees(50);
		}
		else
			throw new Exception("Unknown classifier scheme: "+baseClassifierType);
		if(!baseClassifierOptions.isEmpty())
			classifier.setOptions(weka.core.Utils.splitOptions(baseClassifierOptions));
		if(!rebalanceScheme.isEmpty())
			classifier = new ClassRebalance(classifier, weka.core.Utils.splitOptions(rebalanceScheme));
		return classifier;
	}
	
	/**
	 * <h1>convertScheme</h1>
	 * @param abbr the scheme's abbreviation
	 * @return an option string (convertible through {@link weka.core.Utils#splitOptions}) that can be
	 * used to send the scheme's options to {@link algorithms.rebalance.ClassRebalance}
	 */
	public String convertScheme(String abbr) {
		abbr = abbr.toLowerCase();
		String ret = "";
		if(abbr.startsWith("d")) {
			ret += "-debug ";
			abbr = abbr.substring(1);
		}
		if(!abbr.startsWith("u"))
			ret += "-sensitive 0.1 ";
		else {
			ret += "-sensitive 0 ";
			abbr = abbr.substring(1);
		}
		
		if(abbr.startsWith("t") && !abbr.startsWith("th")) {
			ret += "-rebalance tune ";
			abbr = abbr.substring(1);
		}
		else if(abbr.startsWith("1") || abbr.startsWith("e") || abbr.startsWith("m")|| abbr.startsWith("g")) {
			ret += "-rebalance 1 ";
			if(abbr.startsWith("1"))
				abbr = abbr.substring(1);
		}
		else if(abbr.startsWith("0")) {
			ret += "-rebalance 0 ";
			abbr = abbr.substring(1);
		}
		
		if(abbr.startsWith("e") && !abbr.startsWith("exp")) {
			ret += "-dynamic entropy ";
			abbr = abbr.substring(1);
		}
		else if(abbr.startsWith("m")) {
			ret += "-dynamic max ";
			abbr = abbr.substring(1);
		}
		else if(abbr.startsWith("g")) {
			ret += "-dynamic margin ";
			abbr = abbr.substring(1);
		}
		else
			ret += "-dynamic none ";
		
		if(abbr.startsWith("inv")) {
			ret += "-function inv ";
			abbr = abbr.substring(3);
		}
		else if(abbr.startsWith("exp")){
			ret += "-function exp ";
			abbr = abbr.substring(3);
		}
		else if(abbr.startsWith("lin")){
			ret += "-function lin ";
			abbr = abbr.substring(3);
		}
		else if(abbr.startsWith("thr")){
			ret += "-function threshold ";
			abbr = abbr.substring(3);
		}
		else if(abbr.startsWith("log")){
			ret += "-function log ";
			abbr = abbr.substring(3);
		}
		else
			ret += "-function none ";
		
		if(abbr.endsWith("sample"))
			ret += "-preprocess resample";
		else if(abbr.endsWith("smote"))
			ret += "-preprocess SMOTE";
		else
			ret += "-preprocess none";
		return ret;
	}
	
	@Override
	public String toString() {
		return name;
	}
}
