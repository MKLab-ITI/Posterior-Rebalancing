import java.util.ArrayList;

import algorithms.rebalance.ExperimentScheme;
import importer.DatasetScheme;

/**
 * <h1>TestClassifier</h1>
 * The {@link #main(String[])} method tests a single classifier.
 * @author Emmanouil Krasanakis
 */
public class TestClassifier {
	public static void main(String[] args) throws Exception {
		// SELECT METRICS
		ArrayList<EvaluationMetric> metrics = new ArrayList<EvaluationMetric>();
		metrics.add(new EvaluationMetric.GM());
		metrics.add(new EvaluationMetric.Imbalance());
		metrics.add(new EvaluationMetric.AUC());
		metrics.add(new EvaluationMetric.ILoss());
		// IMPORT INSTANCES
		ExperimentScheme baseScheme = new ExperimentScheme("Logistic", "", "0");
		ExperimentScheme experimentScheme = new ExperimentScheme("Logistic", "", "D0+Sample");//use a scheme that shows debug (D) outputs
		DatasetScheme datasetScheme = new DatasetScheme("data/UCI/car.data", ".data", 10, false);
		System.out.println(GenerateLatexResults.latexResults("\t", datasetScheme, experimentScheme, baseScheme, metrics));
		// NOTIFY USER
		java.awt.Toolkit.getDefaultToolkit().beep();
	}
}
