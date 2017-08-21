import algorithms.rebalance.ExperimentScheme;
import importer.DatasetScheme;

public class Main {
	public static void main(String[] args) throws Exception {
		//IMPORT INSTANCES
		ExperimentScheme experimentScheme = new ExperimentScheme("Adaptive", "", "0+Sample");
		DatasetScheme datasetScheme = new DatasetScheme("data/UCI/car.data", ".data", 10, false);
		System.out.println(GenerateLatexResults.latexResults("\t", datasetScheme, experimentScheme, experimentScheme));
		java.awt.Toolkit.getDefaultToolkit().beep();
	}
}
