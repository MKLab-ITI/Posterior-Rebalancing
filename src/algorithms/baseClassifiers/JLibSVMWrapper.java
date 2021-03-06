package algorithms.baseClassifiers;
import java.io.Serializable;
import java.util.HashSet;

import edu.berkeley.compbio.jlibsvm.ImmutableSvmParameter;
import edu.berkeley.compbio.jlibsvm.ImmutableSvmParameterGrid;
import edu.berkeley.compbio.jlibsvm.binary.BinaryModel;
import edu.berkeley.compbio.jlibsvm.binary.C_SVC;
import edu.berkeley.compbio.jlibsvm.binary.MutableBinaryClassificationProblemImpl;
import edu.berkeley.compbio.jlibsvm.kernel.GaussianRBFKernel;
import edu.berkeley.compbio.jlibsvm.kernel.KernelFunction;
import edu.berkeley.compbio.jlibsvm.util.SparseVector;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

/**
 * <h1>LibSVMWrapper</h1>
 * This class can be used to support the Java LibSVM implementation for Support Vector Machines as a Weka
 * {@link Classifier}.
 * 
 * @author Emmanouil Krasanakis
 */
public class JLibSVMWrapper extends Classifier implements Serializable {
	private static final long serialVersionUID = 941746327712873734L;
	protected int classIndex;
	protected transient  BinaryModel<Integer, SparseVector> model;
	
	public JLibSVMWrapper() {
	}

	@Override
	public void buildClassifier(Instances instances) throws Exception {
		C_SVC<Integer, SparseVector> svm = new C_SVC<Integer, SparseVector>();
		ImmutableSvmParameterGrid.Builder<Integer, SparseVector> builder = ImmutableSvmParameterGrid.builder();

	    // create training parameters ------------
	    HashSet<Float> cSet = new HashSet<Float>();
	    HashSet<KernelFunction<SparseVector>> kernelSet = new HashSet<KernelFunction<SparseVector>>();
	    cSet.add(1.0f);
	    kernelSet.add(new GaussianRBFKernel(1f));

	    // configure finetuning parameters
	    builder.eps = 0.001f; // epsilon
	    builder.Cset = cSet; // C values used
	    builder.kernelSet = kernelSet; //Kernel used

	    ImmutableSvmParameter<Integer, SparseVector> params = builder.build();
	    
		if(instances.classAttribute().numValues()!=2)
			throw new RuntimeException("Only binary problems supported");
        classIndex = instances.classIndex();
	    MutableBinaryClassificationProblemImpl<Integer, SparseVector> problem = new MutableBinaryClassificationProblemImpl<Integer, SparseVector>(String.class, instances.numInstances());
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
