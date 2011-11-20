package mulan.classifier.ensemble;

import java.lang.Math;
import java.util.Arrays;

import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelOutput;

public class TopKEnsemble extends MultiLabelEnsembleLearner {

  private int numScoresToRegard;

  public TopKEnsemble(MultiLabelLearner[] theClassifiers, int theNumScoresToRegard) {
    super(theClassifiers);
    numScoresToRegard = theNumScoresToRegard;
  }

  public TopKEnsemble(MultiLabelLearner[] theClassifiers) {
    this(theClassifiers, 3);
  }


  public MultiLabelOutput makePredictionInternal
                              (MultiLabelOutput[] classifierOutputs) {

    double[][] scoreVectors = new double[classifiers.length][];
    for (int i = 0; i < classifierOutputs.length; i++)
      scoreVectors[i] = classifierOutputs[i].getConfidences();

    double[] scoreVector = new double[scoreVectors[0].length];
    for (int i = 0; i < scoreVector.length; i++) {

      double[] labelScores = new double[scoreVectors.length];
      for (int j = 0; j < scoreVectors.length; j++)
        labelScores[j] = scoreVectors[j][i];

      Arrays.sort(labelScores);
      scoreVector[i] = 0;
      int K = Math.min(numScoresToRegard, labelScores.length);
      for (int k = 1; k <= K; k++)
        scoreVector[i] += labelScores[labelScores.length - k];
      scoreVector[i] /= K;
    }

    return new MultiLabelOutput(scoreVector);
  }
}
