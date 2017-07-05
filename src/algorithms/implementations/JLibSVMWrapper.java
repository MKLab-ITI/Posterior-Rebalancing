package algorithms.implementations;
import java.io.Serializable;
import java.util.HashSet;

import edu.berkeley.compbio.jlibsvm.ImmutableSvmParameter;
import edu.berkeley.compbio.jlibsvm.ImmutableSvmParameterGrid;
import edu.berkeley.compbio.jlibsvm.binary.BinaryModel;
import edu.berkeley.compbio.jlibsvm.binary.C_SVC;
import edu.berkeley.compbio.jlibsvm.binary.MutableBinaryClassificationProblemImpl;
import edu.berkeley.compbio.jlibsvm.kernel.LinearKernel;
import edu.berkeley.compbio.jlibsvm.util.SparseVector;
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
public class JLibSVMWrapper extends Classifier implements Serializable {
	private static final long serialVersionUID = 941746327712873734L;
	protected int classIndex;
	protected transient  BinaryModel model;
	
	public JLibSVMWrapper() {
	}

	@Override
	public void buildClassifier(Instances instances) throws Exception {
		C_SVC svm = new C_SVC();
		ImmutableSvmParameterGrid.Builder builder = ImmutableSvmParameterGrid.builder();

	    // create training parameters ------------
	    HashSet<Float> cSet;
	    HashSet kernelSet;

	    cSet = new HashSet<Float>();
	    cSet.add(1.0f);

	    kernelSet = new HashSet();
	    kernelSet.add(new edu.berkeley.compbio.jlibsvm.kernel.GaussianRBFKernel(1f));

	    // configure finetuning parameters
	    builder.eps = 0.001f; // epsilon
	    builder.Cset = cSet; // C values used
	    builder.kernelSet = kernelSet; //Kernel used

	    ImmutableSvmParameter params = builder.build();
	    
		if(instances.classAttribute().numValues()!=2)
			throw new RuntimeException("Only binary problems supported");
        classIndex = instances.classIndex();
	    MutableBinaryClassificationProblemImpl problem = new MutableBinaryClassificationProblemImpl(String.class, instances.numInstances());
        for(int i=0;i<instances.numInstances();i++) {
        	int classValue = (int)instances.instance(i).value(classIndex)==0?-1:1;
    	    problem.addExample(generateFeatures(instances.instance(i), classIndex), classValue);
        }
	    model = svm.train(problem, params);
	}
	
	@Override
	public double[] distributionForInstance(Instance instance) throws Exception {
		SparseVector xTest = generateFeatures(instance, classIndex);
	    double predictedLabel = model.predictValue(xTest);
	    predictedLabel = 1/(1+Math.exp(-predictedLabel));
	    double[] ret = new double[2];
	    ret[0] = predictedLabel;
	    ret[1] = 1-predictedLabel;
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
	
	private static SparseVector generateFeatures(Instance instance, int classIndex) {
	    SparseVector sparseVector = new SparseVector(instance.numAttributes());
	    int[] indices = new int[instance.numAttributes()];
	    float[] values = new float[instance.numAttributes()];
	    for (int i = 0; i < classIndex; i++) {
	    	indices[i] = new Integer(i);
	    	values[i] = (float) instance.value(i);
	    }
	    for (int i = classIndex+1; i < instance.numAttributes(); i++) {
	    	indices[i-1] = new Integer(i-1);
	    	values[i-1] = (float) instance.value(i);
	    }
	    sparseVector.indexes = indices;
	    sparseVector.values = values;
	    return sparseVector;
	  }
}
