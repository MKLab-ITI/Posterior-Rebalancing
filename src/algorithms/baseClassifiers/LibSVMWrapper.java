package algorithms.baseClassifiers;
import java.io.Serializable;

import libsvm.svm;
import libsvm.svm_model;
import libsvm.svm_node;
import libsvm.svm_parameter;
import libsvm.svm_problem;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

/**
 * <h1>LibSVMWrapper</h1>
 * This class can be used to support the libsvm implementation for Support Vector Machines as a Weka
 * {@link Classifier}.
 * 
 * @author Krasanakis Emmanouil
 */
public class LibSVMWrapper extends Classifier implements Serializable {
	private static final long serialVersionUID = 941746327712873734L;
	protected svm_model model;
	protected svm_parameter param;
	protected int classIndex;
	
	public LibSVMWrapper() {
	}
	/**
	 * <h1>toInputNodes</h1>
	 * Convert a {@link weka.core.Instance} object to an array of {@link libsvm.svm_node} structures.
	 * @param instance
	 * @return
	 */
	protected svm_node[] toInputNodes(Instance instance) {
		svm_node[] node = new svm_node[instance.numAttributes()-1]; 
		for(int m=0;m<classIndex;m++) {
			node[m] = new svm_node();
			node[m].index = m;
			node[m].value = instance.value(m);
			if(Double.isNaN(node[m].value))
				node[m].value = 0;
		}
		for(int m=classIndex;m<node.length;m++) {
			node[m] = new svm_node();
			node[m].index = m;
			node[m].value = instance.value(m+1);
			if(Double.isNaN(node[m].value))
				node[m].value = 0;
		}
		return node;
	}

	@Override
	public void buildClassifier(Instances instances) throws Exception {
		param = new svm_parameter();
		param.probability = 1;
        param.svm_type = svm_parameter.C_SVC;
        param.kernel_type = svm_parameter.LINEAR;
        param.gamma = 1.0/(instances.numAttributes()-1);
        param.nu = 0.5;
        param.cache_size = 100;
        param.C = 1;
        param.eps = 0.001;
        param.p = 0.1;
        
        classIndex = instances.classIndex();
		svm_problem problem = new svm_problem();
		problem.l = instances.numInstances();
		problem.x = new svm_node[problem.l][];
		problem.y = new double[problem.l];
		for(int s=0;s<problem.l;s++) {
			Instance instance = instances.instance(s);
			problem.x[s] = toInputNodes(instance);
			problem.y[s] = (int)instance.classValue();
		} 
		model = svm.svm_train(problem, param);
	}
	
	@Override
	public double[] distributionForInstance(Instance instance) throws Exception {
		int[] labels = new int[svm.svm_get_nr_class(model)];
		double[] probs = new double[svm.svm_get_nr_class(model)];
		svm.svm_get_labels(model,labels);
		svm_node[] x = toInputNodes(instance);
		svm.svm_predict_probability(model,x,probs);
		int maxLabel = 0;
		for(int i=0;i<labels.length;i++) 
			if(labels[i]>maxLabel)
				maxLabel = labels[i];
		double[] ret = new double[maxLabel+1];
		for(int i=0;i<labels.length;i++) 
			ret[labels[i]] = probs[i];
		return ret;
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
}
