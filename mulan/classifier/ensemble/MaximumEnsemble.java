package mulan.classifier.ensemble;

import java.lang.Double;

import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelOutput;

public class MaximumEnsemble extends MultiLabelEnsembleLearner {

  public MaximumEnsemble(MultiLabelLearner[] theClassifiers) {
    super(theClassifiers);
  }


  public MultiLabelOutput makePredictionInternal
                              (MultiLabelOutput[] classifierOutputs) {

    double[][] scoreVectors = new double[classifiers.length][];
    for (int i = 0; i < classifierOutputs.length; i++)
      scoreVectors[i] = classifierOutputs[i].getConfidences();

    double[] scoreVector = new double[scoreVectors[0].length];
    for (int i = 0; i < scoreVector.length; i++) {
      scoreVector[i] = Double.NEGATIVE_INFINITY;
      for (int j = 0; j < scoreVectors.length; j++)
        if (scoreVectors[j][i] > scoreVector[i])
          scoreVector[i] = scoreVectors[j][i];
    }

    return new MultiLabelOutput(scoreVector);
  }
}
