package mulan.classifier.ensemble;

import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelOutput;

public class MeanEnsemble extends MultiLabelEnsembleLearner {

  public MeanEnsemble(MultiLabelLearner[] theClassifiers) {
    super(theClassifiers);
  }


  public MultiLabelOutput makePredictionInternal
                              (MultiLabelOutput[] classifierOutputs) {

    double[][] scoreVectors = new double[classifiers.length][];
    for (int i = 0; i < classifierOutputs.length; i++)
      scoreVectors[i] = classifierOutputs[i].getConfidences();

    double[] scoreVector = new double[scoreVectors[0].length];
    for (int i = 0; i < scoreVector.length; i++) {
      scoreVector[i] = 0;
      for (int j = 0; j < scoreVectors.length; j++)
        scoreVector[i] += scoreVectors[j][i];
      scoreVector[i] /= scoreVectors.length;
    }

    return new MultiLabelOutput(scoreVector);
  }
}
