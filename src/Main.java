import java.util.ArrayList;

import algorithms.rebalance.ExperimentScheme;
import importer.DatasetScheme;

public class Main {
	public static void main(String[] args) throws Exception {
		//METRICS
		ArrayList<EvaluationMetric> metrics = new ArrayList<EvaluationMetric>();
		metrics.add(new EvaluationMetric.GM());
		metrics.add(new EvaluationMetric.Imbalance());
		metrics.add(new EvaluationMetric.AUC());
		metrics.add(new EvaluationMetric.ILoss());
		//IMPORT INSTANCES
		ExperimentScheme experimentScheme = new ExperimentScheme("Adaptive", "", "0+Sample");
		DatasetScheme datasetScheme = new DatasetScheme("data/UCI/car.data", ".data", 10, false);
		System.out.println(GenerateLatexResults.latexResults("\t", datasetScheme, experimentScheme, experimentScheme, metrics));
		java.awt.Toolkit.getDefaultToolkit().beep();
	}
}
